import torch
import clip
from PIL import Image
import numpy as np
import logging
import simplejpeg
from pgvector.psycopg2 import register_vector
import json
import psycopg2
import requests
import uvicorn
logging.basicConfig(level=logging.DEBUG)

#use pgvector-python for vector search

"""
video:
    - video id
    - url
    - json metadata
    - framerate
    - ingest date

frame:
    - video id
    - frame #
    - timestamp in video
    - jpeg bytes
    - image feature vector from CLIP
"""

from vidgear.gears import CamGear
import cv2


def analyze_frame(frame): # want it to be a torch tensor by this point
    if not isinstance(frame, torch.Tensor): 
        frame = torch.from_numpy(frame)
    image_features = model.encode_image(frame).detach().cpu().numpy()
    return image_features

class FrameDb:
    def __init__(self, db_url):
        self.db_conn = psycopg2.connect("host=db dbname=video_db user=postgres password=postgres")
        self.db_conn.set_session(autocommit=True)
        cur = self.db_conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.db_conn.commit()
        register_vector(self.db_conn)
        cur.close()
        logging.info("connected to db")

    def one_time_setup(self):
        # setup db schemas
        # video_metadata: id primary key, metadata jsonb, framerate float, ingest_date timestamptz
        # frames: id primary key, video, frame_num int, timestamp float, jpeg bytea, feats vector
        cur = self.db_conn.cursor()
        sql_query = """
        CREATE TABLE IF NOT EXISTS
        videos (
        id serial PRIMARY KEY,
        metadata jsonb,
        framerate float,
        ingest_date timestamptz,
        origin text,
        unique (origin));

        CREATE TABLE IF NOT EXISTS
        frames (id serial PRIMARY KEY, video integer references videos, timestamp float, jpeg bytea, feats vector(512));

        CREATE INDEX ON frames USING ivfflat (feats);
        """
        cur.execute(sql_query)
        self.db_conn.commit()

        cur.close()
        logging.info("created db schema tables")

    def add_video(self, video_url):
        
        metadata_url = r"https://www.youtube.com//oembed?url=" + video_url + r"&format=json" 
        metadata = json.dumps(requests.get(metadata_url).json())
        logging.info("fetched video metadata")
        # start db transaction here
        cur = self.db_conn.cursor()
        cur.execute("SELECT origin FROM videos WHERE origin = %s", (video_url,))
        if cur.fetchone() is not None:
            logging.info("video already exists in db - skipping")
            return
        stream = CamGear(
                source=video_url,
                stream_mode=True,
                logging=True,
                STREAM_RESOLUTION='360p' # clip resizes to 224x224 anyway
        ).start()
        logging.info("starting youtube video stream")
        framerate = stream.framerate
        frame_num = 0
        next_included_frame = 0
        skip_frames = round(framerate)
        logging.info(f"skip_frames = {skip_frames}")
        video_id = self.add_video_metadata(cur, video_url, metadata, framerate)
        while True:
            frame = stream.read()
            if frame is None: # done
                break
            frame_img = Image.fromarray(frame)
            frame = preprocess(frame_img).unsqueeze(0).to(device)
            if frame_num == next_included_frame: # TODO: replace with fancy content-aware saliency measuring skipping of frames
                logging.info(f'frame {frame_num}: analyzing')
                feats = analyze_frame(frame)
                timestamp = frame_num * (1.0 / framerate) # seconds
                frame_id = self.add_frame(cur, video_id, frame, feats, timestamp)
                next_included_frame += skip_frames
                logging.info(f'skipping until {next_included_frame}')
            frame_num += 1
        # end db tx
        logging.info(f"closing youtube video stream")
        self.db_conn.commit()
        logging.info(f"committed video {video_url} to db")
        cur.close()
        stream.stop()

    def add_video_metadata(self, cur, video_url, metadata, framerate):
        sql_query = """
        insert into 
        videos (metadata, framerate, ingest_date, origin)
        values (%(metadata)s, %(framerate)s, NOW(), %(origin)s)
        on conflict do nothing
        returning id;"""
        cur.execute(sql_query, 
                {"metadata": metadata,
                "framerate": framerate,
                "origin": video_url})
        self.db_conn.commit()
        return cur.fetchone()[0]

    def add_frame(self, cur, video_id, frame, feats, timestamp):
        jpeg_bytes = simplejpeg.encode_jpeg(np.ascontiguousarray(frame.cpu().numpy()[0].astype(np.ubyte).transpose(1,2,0)))
        sql_query = """
        insert into
        frames (video, timestamp, jpeg, feats)
        values (%(video)s, %(timestamp)s, %(jpeg)s, %(feats)s)
        returning id;"""
        cur.execute(sql_query, {
            "video": video_id,
            "jpeg": jpeg_bytes, 
            "feats": feats[0],
            "timestamp": timestamp
        })
        self.db_conn.commit()
        return cur.fetchone()[0]
    
    def search(self, search_query: str, limit = 100):
        global logger
        cur = self.db_conn.cursor()
        tokens = clip.tokenize(search_query).to(device)
        query = model.encode_text(tokens)
        query_vec = query.detach().cpu().numpy()
        query_vec = query_vec.reshape(512,)
        sql_query = """
        select f.timestamp, v.metadata, f.feats <-> %(query_vec)s
        from frames f
        join videos v
        on (f.video = v.id)
        order by feats <-> %(query_vec)s desc
        limit %(limit)s;"""
        cur.execute(sql_query, {"query_vec": query_vec, "limit":limit})
        retval = cur.fetchall()
        logging.info(retval)
        #logging.info(retval[0])
        #images = np.array([simplejpeg.decode_jpeg(retval[i][0][4].tobytes()) for i, rv in enumerate(retval)])
        #images = preprocess(images)
        #img_logits, text_logits = model(images, torch.repeat(query, len(retval), dim=0))
        #probs = img_logits.softmax(dim=-1).cpu().nump()
        logging.info("fetched search results:")
        logging.info(retval)
        #for i, row in enumerate(retval):
        #    retval[i].append({"prob": probs[i]})
        cur.close()
        return retval

    def close(self):
        self.db_conn.close()
import time

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "hello world"}

@app.get("/search")
async def get_query(query: str):
    global logging
    global db
    results = db.search(query)
    logging.info(results)
    return results

@app.get("/add_vid")
async def add_vid(url: str):
    global logging
    global db
    results = db.add_video(url)
    logging.info(results)
    return results


if __name__ == "__main__":
    db = FrameDb(db_url=None)
    db.one_time_setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("fetching CLIP model")
    model, preprocess = clip.load("ViT-B/16", device=device)
    logging.info("fetched.")

    uvicorn.run(app, host="0.0.0.0", port=8000)

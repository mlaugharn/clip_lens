##clip lens

A dockerized tool to search the visual contents of a youtube video by text, using OpenAI's CLIP text/image embeddings. The results are then stored in a Postgres db in a vector format. You can add multiple videos and search through all of them with a single query.

approximately does the following things

```
youtubedl :: YouTube Video URL -> Video

decompose :: Video -> List[Video frame]

watch :: List[Video frame] -> List[CLIP embedding]

add memory :: PostgresDB (CLIP embedding) -> CLIP embedding -> PostgresDB (CLIP Embedding)

sentiment :: String -> CLIP embedding

recollect :: PostgresDB (CLIP Embedding) -> String -> List[(Youtube Video URL, timestamp, Video frame)]
```

There isn't really a UI at the moment so just visit the urls in the code to perform the actions.

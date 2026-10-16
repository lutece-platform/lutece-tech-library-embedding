# lutece-tech-library-embedding

In-process text embeddings for Lutece: ONNX Runtime model, similarity and nearest neighbours.

The model runs in the JVM, through [ONNX Runtime](https://onnxruntime.ai/) and the Hugging Face tokenizer of
[DJL](https://djl.ai/): no external service, no text leaving the application.

## Usage

```xml
<dependency>
    <groupId>fr.paris.lutece.plugins</groupId>
    <artifactId>library-embedding</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</dependency>
<dependency>
    <groupId>fr.paris.lutece.plugins</groupId>
    <artifactId>library-embedding-granite-r2</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</dependency>
```

```java
@Inject
private EmbeddingService _embeddingService;

if ( _embeddingService.isAvailable( ) )
{
    float[] idea = _embeddingService.embedDocument( "Sous-titrer automatiquement les vidéos publiées par la Ville" );
    float[] typed = _embeddingService.embedQuery( "Générer les sous-titres des vidéos" );
    double score = Vectors.similarity( idea, typed );
}
```

- `EmbeddingService` loads the model at the first call. When it cannot, `isAvailable( )` answers `false` and the
  error is logged once: the caller falls back on another search.
- `getModelId( )` is the identifier to store with each vector; vectors of another model must be computed again.
- `Vectors.similarity`, `Vectors.nearest` compare normalized vectors; `Vectors.toBytes` / `fromBytes` store them as
  four little-endian bytes per dimension.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `embedding.model.name` | `granite-r2` | model packaged on the classpath under `fr/paris/lutece/plugins/embedding/models/<name>/` |
| `embedding.model.path` | | directory holding `model.onnx`, `tokenizer.json` and `model.properties`, used instead of the classpath |
| `embedding.threads` | `2` | threads of one encoding |
| `embedding.loadOnStartup` | `false` | `true` loads the model when the application starts, which logs at once whether the model runs on this server |

`model.properties` gives `id`, `dimensions`, `pooling` (`CLS` or `MEAN`), `queryPrefix`, `documentPrefix` and
`maxTokens`.

## Runtime

- ONNX Runtime extracts its native library to `java.io.tmpdir`, which must allow execution.
- The quantized Granite R2 model needs a CPU with AVX2.
- At load, the `lutece.embedding` logger gives the model, the load time, the time of a first encoding, the AVX2
  support, the memory limit of the container, the heap and the temporary directory.

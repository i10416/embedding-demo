//> using dep "com.microsoft.onnxruntime:onnxruntime:1.18.0"
//> using dep "ai.djl:api:0.28.0"
//> using dep "ai.djl.sentencepiece:sentencepiece:0.28.0"
//> using dep "ai.djl.huggingface:tokenizers:0.28.0"
//> using dep "com.lihaoyi::pprint:0.9.0"
//> using dep "com.lihaoyi::os-lib::0.10.2"

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import pprint.*

import java.nio.LongBuffer
import scala.jdk.CollectionConverters.*
import scala.jdk.OptionConverters.*
import scala.util.chaining.*
@main def run =

  val modelName = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
  val tokenizer = HuggingFaceTokenizer.newInstance(modelName)
  val encoding = tokenizer.encode("これはテストです")
  val env: OrtEnvironment = OrtEnvironment.getEnvironment()
  val modelDir =
    os.pwd / os.up / "models" / "onnx" / os.SubPath(modelName)
  val sess: OrtSession =
    env.createSession((modelDir / "model.onnx").toString)
  val (inputIds, attentionMask) =
    (encoding.getIds(), encoding.getAttentionMask())
  val tensorDims = Array(1, encoding.getIds().length.toLong)
  val result = sess.run(
    Map(
      "input_ids" -> OnnxTensor.createTensor(
        env,
        LongBuffer.wrap(inputIds),
        tensorDims
      ),
      "attention_mask" -> OnnxTensor.createTensor(
        env,
        LongBuffer.wrap(attentionMask),
        tensorDims
      )
    ).asJava
  )
  val emb = result
    .get("sentence_embedding")
    .toScala
    .get
    .getValue()
    .asInstanceOf[Array[Array[Float]]]
  pprintln(emb)

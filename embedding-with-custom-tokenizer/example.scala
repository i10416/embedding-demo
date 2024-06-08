//> using dep "com.microsoft.onnxruntime:onnxruntime:1.18.0"
//> using dep "ai.djl:api:0.28.0"
//> using dep "ai.djl.sentencepiece:sentencepiece:0.28.0"
//> using dep "com.worksap.nlp:sudachi:0.7.3"
//> using dep "com.lihaoyi::pprint:0.9.0"
//> using dep "com.lihaoyi::os-lib::0.10.2"

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.bert.WordpieceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.worksap.nlp.sudachi
import com.worksap.nlp.sudachi.DictionaryFactory
import com.worksap.nlp.sudachi.Tokenizer.SplitMode
import pprint.*

import java.nio.LongBuffer
import scala.jdk.CollectionConverters.*
import scala.jdk.OptionConverters.*
import scala.util.chaining.*

@main def run =
  val modelName = "cl-nagoya/sup-simcse-ja-base"
  val modelDir =
    os.pwd / os.up / "models" / "onnx" / os.SubPath(modelName)
  val words = SudachiTokenizer.instance
    .tokenize(SplitMode.C, "これはテストです")
    .asScala
    .map(_.surface())
  val voc =
    DefaultVocabulary
      .builder()
      .addFromTextFile((modelDir / "vocab.txt").toNIO)
      .optUnknownToken("[UNK]")
      .build()
  val wp = WordpieceTokenizer(voc, "[UNK]", 512)
  val subwordEncoded =
    words.flatMap(wp.tokenize.andThen(_.asScala)).map(voc.getIndex)

  val env: OrtEnvironment = OrtEnvironment.getEnvironment()

  val sess: OrtSession =
    env.createSession((modelDir / "model.onnx").toString)
  val tensorDims = Array(1, subwordEncoded.length.toLong)

  val result = sess.run(
    Map(
      "input_ids" -> OnnxTensor.createTensor(
        env,
        LongBuffer.wrap(subwordEncoded.toArray),
        tensorDims
      ),
      "attention_mask" -> OnnxTensor.createTensor(
        env,
        LongBuffer.wrap(Array.fill(subwordEncoded.length)(1)),
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

object SudachiTokenizer {
  lazy val instance = {
    val settings =
      s"""|{
          |  "systemDict" : "system_core.dic",
          |  "oovProviderPlugin" : [
          |    {
          |      "class" : "com.worksap.nlp.sudachi.SimpleOovProviderPlugin",
          |      "oovPOS" : [ "名詞", "普通名詞", "一般", "*", "*", "*"]
          |    }
          |  ]
          |}""".stripMargin
    (new DictionaryFactory)
      .create(
        sudachi.Config.fromJsonString(settings, sudachi.PathAnchor.none())
      )
      .create()
  }
}

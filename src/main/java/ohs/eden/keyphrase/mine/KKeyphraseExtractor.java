package ohs.eden.keyphrase.mine;

import java.io.File;
import java.util.List;

import de.bwaldvogel.liblinear.Linear;
import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MultiToken;
import ohs.types.generic.Counter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class KKeyphraseExtractor {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> phrsBiases = Generics.newCounter();

		{
			List<String> lines = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/kphrs.txt");
			phrsBiases = Generics.newCounter(lines.size());

			for (String line : lines) {
				String[] ps = line.split("\t");
				String phrs = ps[0];
				// MSentence sent = MSentence.newSentence(phrs);
				//
				// List<String> l = Generics.newArrayList();
				//
				// for (MultiToken mt : sent) {
				// for (Token t : mt) {
				// l.add(String.format("%s_/_%s", t.get(TokenAttr.WORD), t.get(TokenAttr.POS)));
				// }
				// }
				//
				// phrs = StrUtils.join(" ", l);

				double weight = Double.parseDouble(ps[1]);
				phrsBiases.setCount(phrs, weight);
			}
		}

		DocumentCollection dc = new DocumentCollection(KPPath.COL_DC_DIR);

		List<String> ins = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");

		KPhraseRanker ranker = new KPhraseRanker(dc, null, phrsBiases);
		KPhraseNumberPredictor predictor = new KPhraseNumberPredictor(
				DocumentCollection.readVocab(KPPath.KP_DIR + "ext/vocab_num_pred.ser"),
				Linear.loadModel(new File(KPPath.KP_DIR + "ext/model_num_pred.txt")));

		KKeyphraseExtractor ext = new KKeyphraseExtractor(predictor, ranker);

		for (String line : ins) {
			String[] ps = line.split("\t");
			ps = StrUtils.unwrap(ps);
			List<String> ansPhrss = Generics.newArrayList();

			for (String phrs : ps[0].split(StrUtils.LINE_REP)) {
				ansPhrss.add(phrs);
			}

			String title = ps[1].replace(StrUtils.LINE_REP, "\n");
			String body = ps[2].replace(StrUtils.LINE_REP, "\n");
			String content = title + "\n" + body;
			MDocument doc = MDocument.newDocument(content);
			Counter<MultiToken> phrss = ranker.rank(doc);

			System.out.println(phrss.toStringSortedByValues(true, true, phrss.size(), "\t"));

			int pred_phrs_size = predictor.predict(doc);

			System.out.println();

			// ke.extractTFIDF(content);
		}

		int dseq = 1000;

		// ke.extract(StrUtils.split(text));

		// ke.extractLM(StrUtils.split(text));

		System.out.println("process ends.");
	}

	private KPhraseNumberPredictor predictor;

	private KPhraseRanker ranker;

	public KKeyphraseExtractor(KPhraseNumberPredictor predictor, KPhraseRanker ranker) {
		this.predictor = predictor;
		this.ranker = ranker;
	}

	public Counter<MultiToken> extract(String content) {
		MDocument doc = MDocument.newDocument(content);
		Counter<MultiToken> ret = ranker.rank(doc);
		int size = predictor.predict(doc);
		ret.keepTopNKeys(size);
		return ret;
	}

}

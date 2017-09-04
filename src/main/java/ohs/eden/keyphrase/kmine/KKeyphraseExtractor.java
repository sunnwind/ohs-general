package ohs.eden.keyphrase.kmine;

import java.io.File;
import java.util.List;

import de.bwaldvogel.liblinear.Linear;
import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.ir.eval.Metrics;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
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

		List<String> outs = Generics.newArrayList(ins.size());

		TextFileWriter writer = new TextFileWriter(KPPath.KP_DIR + "ext/samples.txt");

		double ans_cnt = 0;
		double cor_cnt = 0;
		double pred_cnt = 0;

		for (int i = 0; i < ins.size(); i++) {
			// if (i == 500) {
			// break;
			// }

			String line = ins.get(i);
			String[] ps = line.split("\t");
			ps = StrUtils.unwrap(ps);
			List<String> ansPhrss = Generics.newArrayList();

			for (String phrs : ps[0].split(StrUtils.LINE_REP)) {
				MSentence ms = MSentence.newSentence(phrs);

				MultiToken mt = new MultiToken();

				mt.addAll(ms.getTokens());

				ansPhrss.add(mt.toString());
			}

			String title = ps[1].replace(StrUtils.LINE_REP, "\n");
			String body = ps[2].replace(StrUtils.LINE_REP, "\n");
			String content = title + "\n" + body;
			MDocument doc = MDocument.newDocument(content);
			Counter<MultiToken> preds = ranker.rank(doc);

			// System.out.println(phrss.toStringSortedByValues(true, true, phrss.size(),
			// "\t"));

			int pred_phrs_size = predictor.predict(doc);

			preds.keepTopNKeys(pred_phrs_size);

			ans_cnt += ansPhrss.size();
			pred_cnt += preds.size();

			for (MultiToken pred : preds.keySet()) {
				String s = pred.toString();
				if (ansPhrss.contains(s)) {
					cor_cnt++;
				}
			}

			// Counter<String> c = Generics.newCounter();
			// c.incrementAll(ansPhrss, 1);
			// c.normalize();

			String s1 = "";

			{
				StringBuffer sb = new StringBuffer();
				for (MSentence ms : doc) {
					for (MultiToken mt : ms) {
						for (Token t : mt) {
							sb.append(t.get(0));
						}
						sb.append(" ");
					}
					sb.append("\n");
				}
				s1 = sb.toString().trim();
			}

			String s2 = "";

			{
				List<String> res = Generics.newArrayList();
				for (MultiToken mt : preds.getSortedKeys()) {
					double score = preds.getCount(mt);
					StringBuffer sb = new StringBuffer();
					for (Token t : mt) {
						sb.append(t.get(0));
					}
					res.add(String.format("%s\t%f", sb.toString(), score));
				}
				s2 = StrUtils.join("\n", res);
			}

			writer.write(String.format("No:\t%d", i + 1));
			writer.write(String.format("\n<Input>:\n%s", s1.toString()));
			writer.write(String.format("\n<Output>:\n%s\n\n", s2));

			int prog = BatchUtils.progress(i + 1, ins.size());
			;

			if (prog > 0) {
				double precision = cor_cnt / pred_cnt;
				double recall = cor_cnt / ans_cnt;
				double f1 = Metrics.f1(precision, recall);

				System.out.printf("[%d percent, %d/%d, %f, %f, %f]\n", prog, i + 1, ins.size(), precision, recall, f1);
			}
		}

		double precision = cor_cnt / pred_cnt;
		double recall = cor_cnt / ans_cnt;
		double f1 = Metrics.f1(precision, recall);

		System.out.printf("answers:\t%f\n", ans_cnt);
		System.out.printf("predictions:\t%f\n", pred_cnt);
		System.out.printf("correct:\t%f\n", cor_cnt);

		System.out.printf("precision:\t%f\n", precision);
		System.out.printf("recall:\t%f\n", recall);
		System.out.printf("f1:\t%f\n", f1);

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

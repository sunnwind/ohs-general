package ohs.eden.keyphrase.kmine;

import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Enumeration;
import java.util.List;
import java.util.Scanner;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import de.bwaldvogel.liblinear.Linear;
import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.ir.eval.Metrics;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.Token;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
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

		// run1();
		run2();

		System.out.println("process ends.");
	}

	public static void run1() throws Exception {
		Counter<String> phrsPats = FileUtils.readStringCounterFromText(KPPath.KP_DIR + "ext/phrs_pat.txt");
		phrsPats.pruneKeysBelowThreshold(2);

		Vocab featIdxer = DocumentCollection.readVocab(KPPath.KP_DIR + "ext/vocab_num_pred.ser");

		KPhraseRanker pr = new KPhraseRanker(featIdxer,
				KCandidatePhraseSearcher.newCandidatePhraseSearcher2(phrsPats.keySet()));

		KPhraseNumberPredictor pnp = new KPhraseNumberPredictor(featIdxer,
				Linear.loadModel(new File(KPPath.KP_DIR + "ext/model_num_pred.txt")));

		KKeyphraseExtractor ext = new KKeyphraseExtractor(pnp, pr);

		List<String> ins = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");

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
			List<String> ps = StrUtils.split("\t", line);

			List<String> anss = Generics.newArrayList();

			String cn = ps.get(0);

			MDocument md1 = MDocument.newDocument(ps.get(1));

			for (MSentence ms : md1) {
				StringBuffer sb = new StringBuffer();
				for (Token t : ms.getTokens()) {
					sb.append(t.get(0) + " ");
				}
				anss.add(sb.toString().trim());
			}

			MDocument md2 = MDocument.newDocument(ps.get(2));

			StringBuffer sb = new StringBuffer();

			for (MSentence ms : md2) {
				StringBuffer sb2 = new StringBuffer();
				for (Token t : ms.getTokens()) {
					sb2.append(t.get(0) + " ");
				}
				sb.append(sb2.toString().trim() + "\n");
			}

			String X = sb.toString().trim();

			Counter<String> preds = ext.extract(ps.get(2));

			writer.write(String.format("No:\t%d", i + 1));
			writer.write(String.format("\n<Input>:\n%s", X));
			writer.write(String.format("\n<Answer>:\n%s", anss.toString()));
			writer.write(String.format("\n<Output>:\n%s\n\n", preds.toString(preds.size())));

			int prog = BatchUtils.progress(i + 1, ins.size());

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
	}

	public static void run2() throws Exception {
		List<String> testData = Generics.newLinkedList();

		{
			ZipFile zipFile = new ZipFile(new File(KPPath.KP_DIR + "ext/test/contents.zip"));

			Enumeration<? extends ZipEntry> entries = zipFile.entries();

			while (entries.hasMoreElements()) {
				ZipEntry entry = entries.nextElement();

				if (entry.isDirectory()) {
					continue;
				}

				String cn = entry.getName().split("[\\./]+")[1];

				if (cn.equals("txt")) {
					continue;
				}

				InputStream stream = zipFile.getInputStream(entry);
				InputStreamReader reader = new InputStreamReader(stream, "UTF-8");
				Scanner inputStream = new Scanner(reader);

				StringBuffer sb = new StringBuffer();

				while (inputStream.hasNext()) {
					String data = inputStream.nextLine(); // Gets a whole line
					sb.append(data + "\n");
				}

				String s = sb.toString().trim();
				MDocument md = MDocument.newDocument(s);
				testData.add(String.format("%s\t%s", cn, md.toString().replace("\n", "<nl>")));

				inputStream.close();
				stream.close();

			}
			zipFile.close();
		}

		Counter<String> phrsPats = FileUtils.readStringCounterFromText(KPPath.KP_DIR + "ext/phrs_pat.txt");

		{
			Counter<String> c = Generics.newCounter();
			for (String pat : phrsPats.keySet()) {
				double cnt = phrsPats.getCount(pat);
				if (cnt < 3) {
					continue;
				}

				List<String> poss = StrUtils.split(pat);

				if (!poss.get(poss.size() - 1).startsWith("NN")) {
					continue;
				}

				if (poss.get(0).startsWith("J")) {
					continue;
				}

				c.setCount(pat, cnt);
			}
			phrsPats = c;
		}

		Vocab featIdxer = DocumentCollection.readVocab(KPPath.KP_DIR + "ext/vocab_num_pred.ser");

		KPhraseRanker pr = new KPhraseRanker(featIdxer,
				KCandidatePhraseSearcher.newCandidatePhraseSearcher2(phrsPats.keySet()));

		KPhraseNumberPredictor pnp = new KPhraseNumberPredictor(featIdxer,
				Linear.loadModel(new File(KPPath.KP_DIR + "ext/model_num_pred.txt")));

		KKeyphraseExtractor ext = new KKeyphraseExtractor(pnp, pr);

		TextFileWriter writer = new TextFileWriter(KPPath.KP_DIR + "ext/test_res.txt");

		double ans_cnt = 0;
		double cor_cnt = 0;
		double pred_cnt = 0;

		for (int i = 0; i < testData.size(); i++) {
			// if (i == 500) {
			// break;
			// }

			String line = testData.get(i);
			List<String> ps = StrUtils.split("\t", line);

			String cn = ps.get(0);

			MDocument md2 = MDocument.newDocument(ps.get(1));

			StringBuffer sb = new StringBuffer();

			for (MSentence ms : md2) {
				StringBuffer sb2 = new StringBuffer();
				for (Token t : ms.getTokens()) {
					sb2.append(t.get(0) + " ");
				}
				sb.append(sb2.toString().trim() + "\n");
			}

			String X = sb.toString().trim();

			Counter<String> preds = ext.extract(ps.get(1));

			writer.write(String.format("No:\t%d", i + 1));
			writer.write(String.format("\nCN:\t%s", cn));
			writer.write(String.format("\n<Input>:\n%s", X));
			writer.write(
					String.format("\n<Output>:\n%s\n\n", preds.toStringSortedByValues(true, true, preds.size(), "\t")));

			int prog = BatchUtils.progress(i + 1, testData.size());

			if (prog > 0) {
				double precision = cor_cnt / pred_cnt;
				double recall = cor_cnt / ans_cnt;
				double f1 = Metrics.f1(precision, recall);

				System.out.printf("[%d percent, %d/%d, %f, %f, %f]\n", prog, i + 1, testData.size(), precision, recall,
						f1);
			}
		}
		writer.close();

		double precision = cor_cnt / pred_cnt;
		double recall = cor_cnt / ans_cnt;
		double f1 = Metrics.f1(precision, recall);

		System.out.printf("answers:\t%f\n", ans_cnt);
		System.out.printf("predictions:\t%f\n", pred_cnt);
		System.out.printf("correct:\t%f\n", cor_cnt);

		System.out.printf("precision:\t%f\n", precision);
		System.out.printf("recall:\t%f\n", recall);
		System.out.printf("f1:\t%f\n", f1);
	}

	private int cand_size = Integer.MAX_VALUE;

	private KPhraseNumberPredictor predictor;

	private KPhraseRanker ranker;

	public KKeyphraseExtractor(KPhraseNumberPredictor predictor, KPhraseRanker ranker) {
		this.predictor = predictor;
		this.ranker = ranker;
	}

	public Counter<String> extract(MDocument md) {
		Counter<MSentence> phrsScores = ranker.rank(md);

		int size = phrsScores.size();

		if (cand_size > 0 && cand_size < Integer.MAX_VALUE) {
			size = predictor.predict(md);
		}

		Counter<String> ret = Generics.newCounter();

		for (MSentence phrs : phrsScores.getSortedKeys()) {
			double score = phrsScores.getCount(phrs);
			if (ret.size() == size) {
				break;
			}

			StringBuffer sb = new StringBuffer();
			for (Token t : phrs.getTokens()) {
				sb.append(t.get(0) + " ");
			}
			ret.setCount(sb.toString().trim(), score);
		}

		return ret;
	}

	/**
	 * 전북대 형태소 분석기 결과 텍스트
	 * 
	 * 
	 * 
	 * Word POS CHUNK NER (4개의 필드값)
	 * 
	 * 철분 NNG B O
	 * 
	 * 흡수 NNG B O
	 * 
	 * 를 JKO I O
	 * 
	 * 증가 NNG B O
	 * 
	 * 시킬 XSV I O
	 * 
	 * 수 NNB B O
	 * 
	 * 있 VV B O
	 * 
	 * 는 ETM I O
	 * 
	 * 새로운 VA~ETM B O
	 * 
	 * 형태 NNG B O
	 * 
	 * 의 JKG I O
	 * 
	 * 철 NNG B O
	 * 
	 * 나노 NNG B O
	 * 
	 * 입자 NNG I O
	 * 
	 * 
	 * @param content
	 * @return 핵심구과 확률값을 가지고 있는 Counter
	 */
	public Counter<String> extract(String content) {
		return extract(MDocument.newDocument(content));
	}

	public int getCandidateSize() {
		return cand_size;
	}

	public KPhraseNumberPredictor getPredictor() {
		return predictor;
	}

	public KPhraseRanker getRanker() {
		return ranker;
	}

	public void setCandidateSize(int cand_size) {
		this.cand_size = cand_size;
	}

}

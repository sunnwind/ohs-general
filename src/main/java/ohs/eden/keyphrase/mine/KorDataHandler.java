package ohs.eden.keyphrase.mine;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import edu.stanford.nlp.ling.Sentence;
import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.RawDocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.matrix.DenseMatrix;
import ohs.ml.glove.CooccurrenceCounter;
import ohs.ml.glove.GloveModel;
import ohs.ml.glove.GloveParam;
import ohs.ml.glove.GloveTrainer;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class KorDataHandler {

	public static String[] ATTRS = { "type", "cn", "kor_kwds", "eng_kwds", "kor_title", "eng_title", "kor_abs",
			"eng_abs", "kor_pos_kwds", "kor_pos_title", "kor_pos_abs" };

	public static String getString(MSentence sent, TokenAttr attr) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < sent.size(); i++) {
			MultiToken mt = sent.get(i);
			for (int j = 0; j < mt.size(); j++) {
				Token t = mt.get(j);
				sb.append(t.get(attr));
				if (j < mt.size() - 1) {
					// sb.append(MultiToken.DELIM);
					sb.append(" ");
				}
			}

			if (i < sent.size() - 1) {
				// sb.append(MSentence.DELIM_SENT);
				sb.append(" ");
			}
		}
		return sb.toString();
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		KorDataHandler dh = new KorDataHandler();
		// dh.tagPOS();
		// dh.getKeyphraseDocuments();
		// dh.trainGlove();
		// dh.getLabelData();
		// dh.getKeyphrases();
		dh.getKeyphrasePatterns();
		// dh.test();
		System.out.println("process ends.");
	}

	public void getKeyphraseDocuments() throws Exception {
		RawDocumentCollection rdc = new RawDocumentCollection(KPPath.COL_DC_DIR);

		Counter<String> c = Generics.newCounter();

		List<String> res = Generics.newLinkedList();

		for (int i = 0; i < rdc.size(); i++) {
			int prog = BatchUtils.progress(i + 1, rdc.size());
			if (prog > 0) {
				System.out.printf("[%d percent]\n", prog);
			}

			HashMap<String, String> m = rdc.getMap(i);
			List<String> vals = rdc.get(i);
			int j = 0;
			String type = vals.get(j++);
			String cn = vals.get(j++);
			String korKwsStr = vals.get(j++);
			String engKwsStr = vals.get(j++);
			String korTitle = vals.get(j++);
			String engTitle = vals.get(j++);
			String korAbs = vals.get(j++);
			String engAbs = vals.get(j++);
			String korPosKwds = vals.get(j++);
			String korPosTitle = vals.get(j++);
			String korPosAbs = vals.get(j++);

			if (type.equals("patent")) {
				break;
			}

			if (korAbs.length() == 0) {
				continue;
			}

			String[] korKwds = korKwsStr.split("\n");
			String[] engKwds = engKwsStr.split("\n");

			if (korKwds.length > 0 && korKwds.length == engKwds.length) {

			} else {
				continue;
			}

			List<String> korKwds2 = Generics.newArrayList();
			List<String> engKwds2 = Generics.newArrayList();

			for (int k = 0; k < korKwds.length; k++) {
				String korKwd = korKwds[k];
				String engKwd = engKwds[k];

				if (korKwd.length() > 0 && engKwd.length() > 0) {
					korKwds2.add(korKwd);
					engKwds2.add(engKwd);

					String[] p = new String[] { korKwd, engKwd };
					p = StrUtils.wrap(p);
					c.incrementCount(StrUtils.join("\t", p), 1);
				}
			}

			if (korKwds2.size() > 0) {
				List<String> l = Generics.newArrayList();
				l.add(i + "");
				l.add(StrUtils.join(StrUtils.LINE_REP, korKwds2));
				l.add(korTitle + StrUtils.LINE_REP + korAbs);
				l = StrUtils.wrap(l);
				// res.add(StrUtils.join("\t", l));
				res.add(i + "");
			}
		}
		rdc.close();

		FileUtils.writeStringCollectionAsText(KPPath.KP_DIR + "ext/kphrs_dseq.txt", res);
	}

	public void getKeyphrasePatterns() throws Exception {
		List<String> lines = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");

		Counter<String> c = Generics.newCounter();
		CounterMap<String, String> cm = Generics.newCounterMap();

		for (String line : lines) {
			List<String> ps = StrUtils.split("\t", line);

			if (ps.size() != 3) {
				continue;
			}

			ps = StrUtils.unwrap(ps);
			List<String> kwds = StrUtils.split(StrUtils.LINE_REP, ps.get(0));
			String title = ps.get(1);
			String abs = ps.get(2);

			for (String kwd : kwds) {
				MSentence sent = MSentence.newSentence(kwd);

				String wordPat = getString(sent, TokenAttr.WORD);
				String posPat = getString(sent, TokenAttr.POS);

				c.incrementCount(posPat, 1);
				cm.incrementCount(posPat, wordPat, 1);
			}
		}

		FileUtils.writeStringCounterAsText(KPPath.KP_DIR + "ext/kphrs_pat.txt", c);
	}

	public void getKeyphrases() throws Exception {
		List<String> lines = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");

		Counter<String> c = Generics.newCounter();

		for (String line : lines) {
			List<String> ps = StrUtils.split("\t", line);

			if (ps.size() != 3) {
				continue;
			}

			ps = StrUtils.unwrap(ps);
			List<String> kphrss = StrUtils.split(StrUtils.LINE_REP, ps.get(0));
			String title = ps.get(1);
			String abs = ps.get(2);

			for (String phrs : kphrss) {
				MSentence sent = MSentence.newSentence(phrs);

				String wordPat = getString(sent, TokenAttr.WORD);
				String posPat = getString(sent, TokenAttr.POS);

				c.incrementCount(sent.toString(), 1);
			}
		}

		FileUtils.writeStringCounterAsText(KPPath.KP_DIR + "ext/kphrs.txt", c);
	}

	public void getLabelData() throws Exception {
		RawDocumentCollection rdc = new RawDocumentCollection(KPPath.COL_DC_DIR);

		List<String> tmp = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/kphrs_dseq.txt");
		List<String> res = Generics.newArrayList(tmp.size());

		for (int i = 0; i < tmp.size(); i++) {
			String r = tmp.get(i);
			int dseq = Integer.parseInt(r);

			List<String> vals = rdc.get(dseq);
			int j = 0;
			String type = vals.get(j++);
			String cn = vals.get(j++);
			String korKwdStr = vals.get(j++);
			String engKwdStr = vals.get(j++);
			String korTitle = vals.get(j++);
			String engTitle = vals.get(j++);
			String korAbs = vals.get(j++);
			String engAbs = vals.get(j++);
			String korPosKwdsStr = vals.get(j++);
			String korPosTitle = vals.get(j++).replace(StrUtils.LINE_REP, "\n");
			String korPosAbs = vals.get(j++).replace(StrUtils.LINE_REP, "\n");
			List<String> korPosKwds = StrUtils.split("\n", korPosKwdsStr);

			if (korPosKwds.size() == 0 || korPosTitle.length() == 0 || korPosAbs.length() == 0) {
				continue;
			}

			String content = korPosTitle + "\n" + korPosAbs;

			List<String> cs = Generics.newArrayList();

			for (String kphrs : korPosKwds) {
				if (content.contains(kphrs)) {
					cs.add(kphrs);
				}
			}

			if (cs.size() == 0) {
				continue;
			}

			String[] subs = { StrUtils.join("\n", cs), vals.get(9), vals.get(10) };

			for (int k = 0; k < subs.length; k++) {
				subs[k] = subs[k].replace("\n", StrUtils.LINE_REP);
			}

			subs = StrUtils.wrap(subs);
			res.add(StrUtils.join("\t", subs));
		}
		rdc.close();

		FileUtils.writeStringCollectionAsText(KPPath.KP_DIR + "ext/label_data.txt", res);
	}

	private String getText(List<List<List<Pair<String, String>>>> result) {
		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < result.size(); i++) {
			List<List<Pair<String, String>>> ll = result.get(i);
			for (int j = 0; j < ll.size(); j++) {
				List<Pair<String, String>> l = ll.get(j);

				for (int k = 0; k < l.size(); k++) {
					Pair<String, String> pair = l.get(k);
					// String f = pair.getFirst().replace(" ", "_");
					String f = pair.getFirst().replace(" ", "");
					String s = pair.getSecond();

					if (s.length() == 0) {
						continue;
					}

					// sb.append(String.format("%s%s%s", f, Token.DELIM, s));
					sb.append(String.format("%s%s%s", f, " / ", s));

					if (k != l.size() - 1) {
						// sb.append("+");
						sb.append(MultiToken.DELIM);
					}
				}

				if (j != ll.size() - 1) {
					// sb.append("\n");
					sb.append(" __ ");
				}
			}
			if (i != ll.size() - 1) {
				sb.append("\n");
			}
		}

		return sb.toString().trim();
	}

	public void tagPOS() throws Exception {
		Komoran komoran = new Komoran("lib/models-full/");

		FileUtils.deleteFilesUnder(KPPath.COL_LINE_POS_DIR);

		for (File file : FileUtils.getFilesUnder(KPPath.COL_LINE_DIR)) {
			List<String> lines = FileUtils.readLinesFromText(file);

			for (int i = 0; i < lines.size(); i++) {
				String line = lines.get(i);

				List<String> parts = Generics.newArrayList(line.split("\t"));

				parts = StrUtils.unwrap(parts);

				String type = parts.get(0);
				String cn = parts.get(1);
				String korKwdStr = parts.get(2);
				String engKwdStr = parts.get(3);
				String korTitle = parts.get(4);
				String engTitle = parts.get(5);
				String korAbs = parts.get(6);
				String engAbs = parts.get(7);

				List<String> korKwds = StrUtils.split(StrUtils.LINE_REP, korKwdStr);
				List<String> engKwds = StrUtils.split(StrUtils.LINE_REP, engKwdStr);

				if (korKwds.size() > 0) {
					List<String> korKwds2 = Generics.newArrayList(korKwds.size());
					for (int j = 0; j < korKwds.size(); j++) {
						String kwd = korKwds.get(j);
						kwd = getText(komoran.analyze(kwd, 1));
						korKwds2.add(kwd);
					}

					parts.add(StrUtils.join(StrUtils.LINE_REP, korKwds2));
				} else {
					parts.add("");
				}

				if (korTitle.length() > 0) {
					List<String> l = Generics.newArrayList();
					for (String sent : korTitle.replace(". ", ".\n").split("\n")) {
						l.add(getText(komoran.analyze(sent, 1)));
					}
					parts.add(StrUtils.join(StrUtils.LINE_REP, l));
				} else {
					parts.add("");
				}

				if (korAbs.length() > 0) {

					List<String> l = Generics.newArrayList();
					for (String sent : korAbs.replace(". ", ".\n").split("\n")) {
						List<List<List<Pair<String, String>>>> rs = komoran.analyze(sent, 1);
						l.add(getText(rs));
					}

					parts.add(StrUtils.join(StrUtils.LINE_REP, l));
				} else {
					parts.add("");
				}

				parts = StrUtils.wrap(parts);

				lines.set(i, StrUtils.join("\t", parts));
			}

			FileUtils.writeStringCollectionAsText(file.getPath().replace("line", "line_pos"), lines);
		}
	}

	public void test() throws Exception {

		{
			RawDocumentCollection dc = new RawDocumentCollection(KPPath.COL_DC_DIR);

			System.out.println(dc.getAttrData());

			for (int i = 0; i < 1000000; i++) {
				String s = dc.getMap(i).toString();
				String s2 = dc.getMap(i).get("kor_pos_abs");

				if (s2.length() > 0) {
					System.out.println(i);
					System.out.println(s2);
					System.out.println("======================");
				}
			}
		}

		// {
		// DocumentCollection dc = new DocumentCollection(KPPath.COL_DC_DIR);
		//
		// String s = dc.getText(32586).getSecond();
		//
		// System.out.println(s);
		// System.out.println("======================");
		// }

	}

	public void test2() throws Exception {
		List<String> lines = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");

		Counter<String> c = Generics.newCounter();

		for (String line : lines) {
			List<String> ps = StrUtils.split("\t", line);

			if (ps.size() != 3) {
				continue;
			}

			ps = StrUtils.unwrap(ps);
			List<String> kwds = StrUtils.split(StrUtils.LINE_REP, ps.get(0));
			String title = ps.get(1);
			String abs = ps.get(2);

			for (String kwd : kwds) {
				MSentence sent = MSentence.newSentence(kwd);

				String wordPat = getString(sent, TokenAttr.WORD);
				String posPat = getString(sent, TokenAttr.POS);

				c.incrementCount(wordPat, 1);
			}
		}

		System.out.println(c.size());
		System.out.println(c.totalCount());

		System.out.println(1d * c.size() / lines.size());
		System.out.println(1d * c.totalCount() / lines.size());

		Counter<Integer> c2 = Generics.newCounter();

		for (String wordPat : c.keySet()) {
			c2.incrementCount(StrUtils.split(wordPat).size(), 1);
		}

		List<Integer> lens = Generics.newArrayList(c2.keySet());
		Collections.sort(lens);

		System.out.println();

		for (int i = 0; i < lens.size(); i++) {
			int len = lens.get(i);
			int cnt = (int) c2.getCount(len);
			System.out.printf("%d\t%d\n", len, cnt);
		}

		// System.out.println(c2.toStringSortedByValues(true, true, c2.size(), "\t"));

	}

	public void trainGlove() throws Exception {
		String dir = KPPath.COL_DIR;
		String scDir = dir + "dc/";
		String ccDir = dir + "cocnt/";
		String outFileName1 = dir + "/emb/glove_model.ser.gz";
		String outFileName2 = dir + "/emb/glove_emb.ser.gz";

		int thread_size = 50;
		int hidden_size = 300;
		int max_iters = 30;
		int window_size = 10;
		double learn_rate = 0.001;
		boolean use_adam = true;
		boolean read_all_files = true;

		int batch_size = 100;

		DocumentCollection dc = new DocumentCollection(scDir);

		Vocab vocab = dc.getVocab();

		if (!FileUtils.exists(ccDir)) {
			CooccurrenceCounter cc = new CooccurrenceCounter(scDir, ccDir, null);
			cc.setWindowSize(window_size);
			cc.setCountThreadSize(thread_size);
			cc.setSymmetric(true);
			cc.setOutputFileSize(batch_size);
			cc.count();
		}

		GloveParam param = new GloveParam(vocab.size(), hidden_size);
		param.setThreadSize(thread_size);
		param.setLearnRate(learn_rate);

		GloveTrainer trainer = new GloveTrainer();
		GloveModel M = trainer.train(param, vocab, ccDir, max_iters, read_all_files, use_adam);
		M.writeObject(outFileName1);

		DenseMatrix E = M.getAveragedModel();
		E.writeObject(outFileName2);

		// DenseMatrix E = new DenseMatrix(KPPath.COL_DIR + "emb/glove_emb.ser.gz");
		RandomAccessDenseMatrix.build(E, KPPath.COL_DIR + "emb/glove_emb_ra.ser");
	}
}

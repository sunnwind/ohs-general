package ohs.eden.keyphrase.ext;

import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.bitbucket.eunjeon.seunjeon.Analyzer;
import org.bitbucket.eunjeon.seunjeon.LNode;
import org.bitbucket.eunjeon.seunjeon.Morpheme;

import kr.co.shineware.util.common.model.Pair;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.RawDocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.io.TextFileWriter;
import ohs.matrix.DenseMatrix;
import ohs.ml.glove.CooccurrenceCounter;
import ohs.ml.glove.GloveModel;
import ohs.ml.glove.GloveParam;
import ohs.ml.glove.GloveTrainer;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import scala.collection.mutable.WrappedArray;

public class KorDataHandler {

	public static String[] ATTRS = { "type", "cn", "kor_kwds", "eng_kwds", "kor_title", "eng_title", "kor_abs",
			"eng_abs", "kor_pos_kwds", "kor_pos_title", "kor_pos_abs" };

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		KorDataHandler dh = new KorDataHandler();
		// dh.tagPOS();
		// dh.getLabelData2();
		// dh.merge();

		// dh.getKeyphraseDocuments();
		// dh.trainGlove();
		// dh.getLabelData();
		// dh.getKeyphrases();
		// dh.getKeyphrasePatterns();

		dh.test();

		System.out.println("process ends.");
	}

	public void test() throws Exception {

		List<String> ins = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");
		List<String> outs = Generics.newLinkedList();
		int doc_cnt = 0;

		for (int i = 0; i < ins.size(); i++) {
			String line = ins.get(i);
			List<String> ps = StrUtils.split("\t", line);

			MDocument d1 = MDocument.newDocument(ps.get(1));
			MDocument d2 = MDocument.newDocument(ps.get(2));

			List<String> kps = Generics.newArrayList();

			for (MSentence s : d1) {
				List<String> ss = s.getTokenStrings(0);
				String kp = StrUtils.join(" ", ss);
				kps.add(kp);
			}

			List<String> ss = Generics.newArrayList();

			for (MSentence s : d2) {
				List<String> t = s.getTokenStrings(0);
				String k = StrUtils.join(" ", t);
				k = k.replace(" . ", " .\n ");
				for (String m : k.split("\n")) {
					ss.add(m.trim());
				}
			}

			String d = StrUtils.join("\n", ss);

			List<String> kps2 = Generics.newArrayList();

			for (String kp : kps) {
				if (d.contains(kp)) {
					kps2.add(kp);
				}
			}

			if (kps2.size() == 0) {
				continue;
			}

			if (doc_cnt == 100) {
				break;
			}

			doc_cnt++;

			String cn = ps.get(0);
			String k = StrUtils.join("<tb>", kps2);

			List<String> res = Generics.newArrayList();
			res.add(cn);
			res.add(k);
			res.add(d.replace("\n", "<nl>"));
			
			res = StrUtils.wrap(res);

			outs.add(StrUtils.join("\t", res));
		}

		FileUtils.writeStringCollectionAsText(KPPath.KP_DIR + "ext/samples.txt", outs);

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
		Counter<String> c1 = FileUtils.readStringCounterFromText(KPPath.KP_DIR + "ext/phrs.txt");
		Counter<String> c2 = Generics.newCounter();

		for (String phrs : c1.keySet()) {
			MDocument md = MDocument.newDocument(phrs);

			for (MSentence ms : md) {
				StringBuffer sb = new StringBuffer();
				for (MToken t : ms) {
					sb.append(t.get(1) + " ");
				}
				String p = sb.toString().trim();
				c2.incrementCount(p, 1);
			}
		}

		FileUtils.writeStringCounterAsText(KPPath.KP_DIR + "ext/phrs_pat.txt", c2);
	}

	public void getKeyphrases() throws Exception {
		Counter<String> c = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt")) {
			List<String> ps = StrUtils.split("\t", line);

			if (ps.size() != 3) {
				continue;
			}

			String cn = ps.get(0);
			MDocument md = MDocument.newDocument(ps.get(1));

			for (MSentence ms : md) {
				c.incrementCount(ms.toString().replace("\n", "<nl>"), 1);
			}
		}

		FileUtils.writeStringCounterAsText(KPPath.KP_DIR + "ext/phrs.txt", c);
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

	public void getLabelData2() throws Exception {
		String[] inFileNames = { "keywords.zip", "contents.zip" };
		String[] outFileNames = { "tmp1.txt", "tmp2.txt" };

		String dirPath = KPPath.KP_DIR + "ext/pos_tagged/";

		for (int i = 0; i < inFileNames.length; i++) {
			ZipFile zipFile = new ZipFile(new File(dirPath + inFileNames[i]));

			Enumeration<? extends ZipEntry> entries = zipFile.entries();

			List<String> res = Generics.newLinkedList();

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
				res.add(String.format("%s\t%s", cn, md.toString().replace("\n", "<nl>")));

				inputStream.close();
				stream.close();

			}
			zipFile.close();

			FileUtils.writeStringCollectionAsText(dirPath + outFileNames[i], res);
		}
	}

	public void merge() throws Exception {
		String[] outFileNames = { "tmp1.txt", "tmp2.txt" };
		String dirPath = KPPath.KP_DIR + "ext/pos_tagged/";

		Map<String, String> m1 = Generics.newHashMap();
		Map<String, String> m2 = Generics.newHashMap();

		for (String line : FileUtils.readLinesFromText(dirPath + outFileNames[0])) {
			String[] ps = line.split("\t");
			m1.put(ps[0], ps[1]);
		}

		for (String line : FileUtils.readLinesFromText(dirPath + outFileNames[1])) {
			String[] ps = line.split("\t");
			m2.put(ps[0], ps[1]);
		}

		List<String> outs = Generics.newArrayList(m1.size());

		for (String cn : m1.keySet()) {
			String title = m1.get(cn);
			String body = m2.get(cn);
			outs.add(String.format("%s\t%s\t%s", cn, title, body));
		}

		FileUtils.writeStringCollectionAsText(KPPath.KP_DIR + "ext/label_data.txt", outs);

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
						sb.append(" ");
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
		FileUtils.deleteFilesUnder(KPPath.COL_LINE_POS_DIR);

		int batch_size = 20000;
		int batch_cnt = 0;

		List<String> outs = Generics.newArrayList(batch_size);

		for (File file : FileUtils.getFilesUnder(KPPath.COL_LINE_DIR)) {
			List<String> ins = FileUtils.readLinesFromText(file);

			for (int i = 0; i < ins.size(); i++) {
				String line = ins.get(i);

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

				MDocument kps = new MDocument();
				MDocument content = new MDocument();

				if (korKwds.size() > 0) {
					for (String kwd : korKwds) {
						MSentence kp = new MSentence();
						for (LNode node : Analyzer.parseJava(kwd)) {
							Morpheme m = node.morpheme();
							WrappedArray<String> fs = m.feature();
							String[] vals = (String[]) fs.array();
							String word = m.surface();
							String pos = vals[0];

							MToken t = new MToken(2);
							t.add(word);
							t.add(pos);
							kp.add(t);
						}
						kps.add(kp);
					}
				}

				if (korTitle.length() > 0 && korAbs.length() > 0) {
					for (String str : (korTitle + "\n" + korAbs).split("\n")) {
						if (str.length() == 0) {
							continue;
						}
						MSentence sent = new MSentence();
						for (LNode node : Analyzer.parseJava(str)) {
							Morpheme m = node.morpheme();
							WrappedArray<String> fs = m.feature();
							String[] vals = (String[]) fs.array();
							String word = m.surface();
							String pos = vals[0];

							MToken t = new MToken(2);
							t.add(word);
							t.add(pos);
							sent.add(t);
						}
						content.add(sent);
					}
				}

				if (kps.size() > 0 && content.size() > 0) {
					List<String> l = Generics.newArrayList();
					l.add(cn);
					l.add(type);
					l.add(content.toString().replace("\n", StrUtils.LINE_REP));
					l = StrUtils.wrap(l);

					outs.add(StrUtils.join("\t", l));
				}

				if (outs.size() > 0 && outs.size() % batch_size == 0) {
					DecimalFormat df = new DecimalFormat("000000");
					String outFileName = KPPath.COL_LINE_POS_DIR + String.format("%s.txt.gz", df.format(batch_cnt++));
					FileUtils.writeStringCollectionAsText(outFileName, outs);
					outs.clear();
				}
			}
		}

		if (outs.size() > 0) {
			DecimalFormat df = new DecimalFormat("000000");
			String outFileName = KPPath.COL_LINE_POS_DIR + String.format("%s.txt.gz", df.format(batch_cnt++));
			FileUtils.writeStringCollectionAsText(outFileName, outs);
			outs.clear();
		}
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

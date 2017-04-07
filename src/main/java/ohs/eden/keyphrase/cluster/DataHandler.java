package ohs.eden.keyphrase.cluster;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Set;

import org.apache.lucene.analysis.en.EnglishAnalyzer;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;
import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.io.TextFileWriter;
import ohs.ir.lucene.common.AnalyzerUtils;
import ohs.ml.glove.CooccurrenceCounter;
import ohs.ml.glove.GloveModel;
import ohs.ml.glove.GloveParam;
import ohs.ml.glove.GloveTrainer;
import ohs.nlp.ling.types.KDocument;
import ohs.nlp.ling.types.KSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.types.common.StrPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.CounterMapMap;
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class DataHandler {

	public static List<String> getKeywords(String keywordStr) {
		List<String> ret = Generics.newArrayList();
		for (String kw : keywordStr.split(";")) {
			kw = kw.trim();
			if (kw.length() > 0) {
				ret.add(kw);
			}
		}
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		DataHandler dh = new DataHandler();
		// dh.tagPOS();

		// dh.extractNouns();
		// dh.createNounCollection();
		// dh.trainGlove();
		// dh.queryGloveModel();

		// dh.createSentenceCollection();
		// dh.trainGlove();

		// dh.makeKeywordData();
		// dh.makeTitleData();
		// dh.makeKeywordDataFromWOS();

		// dh.extractKeywordAbbreviations();

		// dh.getKeywordStats();
		// dh.extractCNs();
		// dh.test();

		System.out.println("process ends.");
	}

	public void test() {
		TextFileReader reader = new TextFileReader(KPPath.KYP_DIR + "keyword_cluster.txt");

		List<String> lines1 = Generics.newArrayList();
		List<String> lines2 = Generics.newArrayList();
		List<String> lines3 = Generics.newArrayList();
		List<String> lines = lines1;

		while (reader.hasNext()) {
			String line = reader.next();

			if (line.startsWith("No:")) {
				String[] parts = line.split("\t");
				int no = Integer.parseInt(parts[1]);
				if (no < 150000) {
					lines = lines1;
				} else if (no < 350000) {
					lines = lines2;
				} else {
					lines = lines3;
				}

			}
			lines.add(line);
		}
		reader.close();

		{

			TextFileWriter writer = new TextFileWriter(KPPath.KYP_DIR + "c1.txt");
			for (String line : lines1) {
				writer.write(line + "\n");
			}
			writer.close();
		}

		{

			TextFileWriter writer = new TextFileWriter(KPPath.KYP_DIR + "c2.txt");
			for (String line : lines2) {
				writer.write(line + "\n");
			}
			writer.close();
		}

		{

			TextFileWriter writer = new TextFileWriter(KPPath.KYP_DIR + "c3.txt");
			for (String line : lines3) {
				writer.write(line + "\n");
			}
			writer.close();
		}
	}

	public void trainGlove() throws Exception {
		String dir = KPPath.COL_DIR;
		String scDir = dir + "dc/";
		String ccDir = dir + "cocnt/";
		String modelFileName = new File(dir).getParent() + "glove_model.ser.gz";

		Set<String> stopwords = Generics.newHashSet();
		stopwords.add("<sto>");

		CooccurrenceCounter cc = new CooccurrenceCounter();
		cc.setOutputFileSize(100);
		cc.setSymmetric(true);
		cc.setCountThreadSize(10);
		cc.setWindowSize(10);
		cc.count(scDir, ccDir, stopwords);

		DocumentCollection lsc = new DocumentCollection(scDir);
		Vocab vocab = lsc.getVocab();

		GloveParam param = new GloveParam(vocab.size(), 300);
		param.setThreadSize(50);
		param.setLearnRate(0.05);

		GloveTrainer trainer = new GloveTrainer();
		GloveModel M = trainer.train(param, vocab, ccDir, 200, true, true);
		M.writeObject(modelFileName);
	}

	public void extractNouns() throws Exception {
		String outDir = KPPath.COL_DIR + "noun/";

		FileUtils.deleteFilesUnder(outDir);

		List<String> res = Generics.newArrayList();
		int batch_size = 20000;
		int batch_cnt = 0;
		DecimalFormat df = new DecimalFormat("000000");

		TextFileReader reader = new TextFileReader(KPPath.COL_DIR + "3p_pos.csv.gz");

		while (reader.hasNext()) {
			if (reader.getLineCnt() == 1) {
				continue;
			}
			String line = reader.next();

			String[] parts = line.split("\t");
			parts = StrUtils.unwrap(parts);

			String type = parts[0];
			String cn = parts[1];
			String korKwdStr = parts[2].split(" => ")[1];

			korKwdStr = StrUtils.unwrap(korKwdStr);
			korKwdStr = korKwdStr.replace(";", "");

			String engKwdStr = parts[3];
			String korTitle = parts[4];
			String engTitle = parts[5];
			String korAbs = parts[6];
			String engAbs = parts[7];

			KDocument d1 = TaggedTextParser.parse(korKwdStr + "\\n" + korTitle + "\\n" + korAbs);
			List<String> ll = Generics.newArrayList();

			for (int i = 0; i < d1.size(); i++) {
				KSentence sent = d1.getSentence(i);

				String[][] words = sent.getSub(TokenAttr.WORD);
				String[][] poss = sent.getSub(TokenAttr.POS);

				List<String> l = Generics.newArrayList();

				for (int j = 0; j < words.length; j++) {
					for (int k = 0; k < words[j].length; k++) {
						if (poss[j][k].startsWith("N")) {
							l.add(words[j][k].trim());
						} else {
							l.add("<sto>");
						}
					}
				}
				ll.add(StrUtils.join(" ", l));
			}

			String[] vals = new String[] { type, cn, StrUtils.join(StrUtils.LINE_REP, ll) };
			vals = StrUtils.wrap(vals);

			res.add(StrUtils.join("\t", vals));

			if (res.size() % batch_size == 0) {
				String outFileName = outDir + String.format("%s.txt.gz", df.format(batch_cnt++));
				FileUtils.writeStringCollectionAsText(outFileName, res);
				res.clear();
			}
		}
		reader.close();

		if (res.size() > 0) {
			String outFileName = outDir + String.format("%s.txt.gz", df.format(batch_cnt++));
			FileUtils.writeStringCollectionAsText(outFileName, res);
		}
	}

	public void extractCNs() throws Exception {
		TextFileWriter writer = new TextFileWriter(KPPath.KYP_DIR + "cn.txt.gz");

		for (File file : new File(KPPath.COL_LINE_DIR).listFiles()) {
			List<String> lines = FileUtils.readLinesFromText(file);

			for (String line : lines) {
				String[] parts = line.split("\t");
				parts = StrUtils.unwrap(parts);

				String type = parts[0];
				String cn = parts[1];
				String korKwdStr = parts[2];
				String engKwdStr = parts[3];
				String korTitle = parts[4].replace("\n", "\\n").replace("\t", "\\t");
				String engTitle = parts[5];
				String korAbs = parts[6].replace("\n", "\\n").replace("\t", "\\t");
				String engAbs = parts[7];

				String[] output = new String[] { cn, type };
				writer.write(StrUtils.join("\t", output) + "\n");
			}
		}
		writer.close();

		// FileUtils.writeStrSetMap(KPPath.KYP_2P_FILE, kwdToDocs);
	}

	public void extractKeywordAbbreviations() throws Exception {
		KeywordData data = new KeywordData();

		if (FileUtils.exists(KPPath.KYP_2P_FILE.replace("txt", "ser"))) {
			data.readObject(KPPath.KYP_2P_FILE.replace("txt", "ser"));
		} else {
			data.add(KPPath.KYP_2P_FILE);
			data.writeObject(KPPath.KYP_2P_FILE.replace("txt", "ser"));
		}

		AbbreviationExtractor ext = new AbbreviationExtractor();

		CounterMap<String, String> cm = Generics.newCounterMap();

		CounterMapMap<String, String, String> dcm = Generics.newCounterMapMap();

		for (StrPair kwdp : data.getKeywordIndexer().getObjects()) {

			for (String kwd : kwdp.asArray()) {
				List<ohs.types.generic.Pair<String, String>> pairs = ext.extract(kwd);

				for (ohs.types.generic.Pair<String, String> pair : pairs) {
					cm.incrementCount(pair.getFirst(), kwdp.getFirst() + "#" + pair.getSecond().toLowerCase(), 1);
					dcm.incrementCount(pair.getFirst(), pair.getSecond().toLowerCase(), kwdp.getFirst(), 1);
				}
			}
		}

		FileUtils.writeStringCounterMapAsText(KPPath.KYP_ABBR_FILE, cm);

		System.out.println(dcm);

	}

	private String getText(List<List<List<Pair<String, String>>>> result) {
		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < result.size(); i++) {
			List<List<Pair<String, String>>> ll = result.get(i);
			for (int j = 0; j < ll.size(); j++) {
				List<Pair<String, String>> l = ll.get(j);

				for (int k = 0; k < l.size(); k++) {
					Pair<String, String> pair = l.get(k);
					String f = pair.getFirst().replace(" ", "_");
					String s = pair.getSecond();

					if (s.length() == 0) {
						continue;
					}

					sb.append(String.format("%s%s%s", f, Token.DELIM_TOKEN, s));
					// sb.append(String.format("%s%s%s", f, "/", s));

					if (k != l.size() - 1) {
						// sb.append("+");
						sb.append(MultiToken.DELIM_MULTI_TOKEN);
					}
				}

				if (j != ll.size() - 1) {
					sb.append("\n");
				}
			}
			if (i != ll.size() - 1) {
				sb.append("\n");
			}
		}

		return sb.toString().trim();
	}

	public void makeKeywordDataFromWOS() throws Exception {
		SetMap<String, String> kwdToDocs = Generics.newSetMap();
		int line_cnt = 0;

		int doc_cnt = 0;
		int doc_cnt_with_kwd = 0;
		int total_kwd_cnt = 0;

		for (String line : FileUtils.readLinesFromText(KPPath.COL_DIR + "wos_keywords.csv")) {
			if (line_cnt++ == 0) {
				continue;
			}
			String[] parts = line.split("\t");
			parts = StrUtils.unwrap(parts);

			String cn = parts[0];
			String engKwdStr = parts[1];

			doc_cnt++;

			int kwd_cnt = 0;

			for (String kwd : engKwdStr.split(";")) {
				kwd = kwd.trim();
				if (kwd.length() == 0) {
					continue;
				}

				kwd_cnt++;
				String[] p = StrUtils.wrap(StrUtils.asArray("", kwd));
				kwdToDocs.put(StrUtils.join("\t", p), cn);
			}

			total_kwd_cnt += kwd_cnt;

			if (kwd_cnt > 0) {
				doc_cnt_with_kwd++;
			}
		}

		System.out.printf("%d/%d\n", doc_cnt_with_kwd, doc_cnt);
		System.out.println(total_kwd_cnt);
		System.out.println(1f * total_kwd_cnt / doc_cnt_with_kwd);

		// FileUtils.writeStrSetMap(KPPath.KYP_WOS_FILE, kwdToDocs);
	}

	public void makeKeywordData() throws Exception {
		SetMap<String, String> kwdToDocs = Generics.newSetMap();

		for (File file : new File(KPPath.COL_LINE_DIR).listFiles()) {
			List<String> lines = FileUtils.readLinesFromText(file);

			for (String line : lines) {
				String[] parts = line.split("\t");
				parts = StrUtils.unwrap(parts);

				String type = parts[0];
				String cn = parts[1];
				String korKwdStr = parts[2].split(" => ")[0];
				String korKwdStrPos = parts[2].split(" => ")[1];
				String engKwdStr = parts[3];
				String korTitle = parts[4];
				String engTitle = parts[5];
				String korAbs = parts[6];
				String engAbs = parts[7];

				if (type.equals("patent")) {
					break;
				}

				korKwdStr = StrUtils.unwrap(korKwdStr);
				korKwdStrPos = StrUtils.unwrap(korKwdStrPos);

				List<String> korKwds = getKeywords(korKwdStr);
				List<String> engKwds = getKeywords(engKwdStr);

				if (korKwds.size() == 0 && engKwds.size() == 0) {

				} else if (korKwds.size() == engKwds.size()) {
					for (int j = 0; j < korKwds.size(); j++) {
						String kor = korKwds.get(j);
						String eng = engKwds.get(j);
						String[] p = StrUtils.wrap(StrUtils.asArray(kor, eng));
						kwdToDocs.put(StrUtils.join("\t", p), cn);
					}
				} else {
					for (String kwd : korKwds) {
						String[] p = StrUtils.wrap(StrUtils.asArray(kwd, ""));
						kwdToDocs.put(StrUtils.join("\t", p), cn);
					}

					for (String kwd : engKwds) {
						String[] p = StrUtils.wrap(StrUtils.asArray("", kwd));
						kwdToDocs.put(StrUtils.join("\t", p), cn);
					}
				}
			}
		}

		FileUtils.writeStringSetMapAsText(KPPath.KYP_2P_FILE, kwdToDocs);
	}

	public void makeTitleData() throws Exception {

		EnglishAnalyzer analyzer = new EnglishAnalyzer();

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (File file : new File(KPPath.COL_LINE_DIR).listFiles()) {
			List<String> lines = FileUtils.readLinesFromText(file);

			for (String line : lines) {
				String[] parts = line.split("\t");

				parts = StrUtils.unwrap(parts);

				String type = parts[0];
				String cn = parts[1];
				String korKwdStr = parts[2].split(" => ")[0];
				String korKwdStrPos = parts[2].split(" => ")[1];
				String engKwdStr = parts[3];
				String korTitle = parts[4];
				String engTitle = parts[5];
				String korAbs = parts[6];
				String engAbs = parts[7];

				if (type.equals("patent")) {
					break;
				}

				korKwdStr = StrUtils.unwrap(korKwdStr);
				korKwdStrPos = StrUtils.unwrap(korKwdStrPos);

				if (korKwdStr.length() == 0 && engKwdStr.length() == 0) {
					continue;
				}

				Counter<String> c = Generics.newCounter();

				KDocument doc = TaggedTextParser.parse(korTitle + "\\n" + korAbs);
				//
				for (Token t : doc.getSubTokens()) {
					String word = t.get(TokenAttr.WORD);
					String pos = t.get(TokenAttr.POS);
					if (pos.startsWith("N")) {
						c.incrementCount(word.toLowerCase(), 1);
					}
				}

				c.incrementAll(AnalyzerUtils.getWordCounts(engTitle, analyzer));

				if (c.size() > 0) {
					cm.setCounter(cn, c);
				}
			}
		}

		FileUtils.writeStringCounterMapAsText(KPPath.KYP_TITLE_DATA_FILE, cm);
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

				// if (!cn.equals("JAKO199910102414471")) {
				// continue;
				// }

				List<String> korKwds = getKeywords(korKwdStr);
				List<String> engKwds = getKeywords(engKwdStr);

				if (korKwds.size() > 0) {
					List<String> korKwds2 = Generics.newArrayList(korKwds.size());
					for (int j = 0; j < korKwds.size(); j++) {
						String kwd = korKwds.get(j);
						kwd = getText(komoran.analyze(kwd, 1));
						korKwds2.add(kwd);
					}

					parts.add(StrUtils.join(";", korKwds2).replace("\n", StrUtils.LINE_REP));
				} else {
					parts.add("");
				}

				if (korTitle.length() > 0) {
					String t = getText(komoran.analyze(korTitle, 1));
					parts.add(t.replace("\n", StrUtils.LINE_REP));
				} else {
					parts.add("");
				}

				if (korAbs.length() > 0) {
					StringBuffer sb = new StringBuffer();
					for (String sent : korAbs.replace(". ", ".\n").split("\n")) {
						sb.append(getText(komoran.analyze(sent, 1)));
						sb.append("\n\n");
					}

					String t = sb.toString().trim();
					parts.add(t.replace("\n", StrUtils.LINE_REP));
				} else {
					parts.add("");
				}

				parts = StrUtils.wrap(parts);

				lines.set(i, StrUtils.join("\t", parts));
			}

			FileUtils.writeStringCollectionAsText(file.getPath().replace("line", "line_pos"), lines);
		}
	}

}

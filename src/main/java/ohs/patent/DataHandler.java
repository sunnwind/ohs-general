package ohs.patent;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Set;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;
import ohs.corpus.type.LargeDocumentCollection;
import ohs.corpus.type.LargeDocumentCollectionCreator;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.eden.keyphrase.cluster.TaggedTextParser;
import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.ml.glove.GloveModel;
import ohs.ml.word2vec.WordSearcher;
import ohs.ml.word2vec.WordVectorModel;
import ohs.nlp.ling.types.KDocument;
import ohs.nlp.ling.types.KSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
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
		// dh.partition();
		// dh.tokenize();
		dh.tagPOS();
		// dh.extractNouns();
		// dh.createNounCollection();
		// dh.trainGlove();
		// dh.queryGloveModel();

		// dh.createSentenceCollection();
		// dh.trainGlove();

		// dh.tagPOS();
		// dh.makeKeywordData();
		// dh.makeTitleData();

		// dh.extractKeywordAbbreviations();

		// dh.getKeywordStats();
		// dh.extractCNs();

		System.out.println("process ends.");
	}

	public void createNounCollection() throws Exception {
		LargeDocumentCollectionCreator ldcc = new LargeDocumentCollectionCreator();
		ldcc.setMinDocFreq(3);
		ldcc.setCountingThreadSize(10);
		// ldcc.create(KPPath.COL_DIR + "noun", new int[] { 2 }, KPPath.COL_DIR + "noun_dc");
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
				FileUtils.writeStringCollection(outFileName, res);
				res.clear();
			}
		}
		reader.close();

		if (res.size() > 0) {
			String outFileName = outDir + String.format("%s.txt.gz", df.format(batch_cnt++));
			FileUtils.writeStringCollection(outFileName, res);
		}
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

					if (k != l.size() - 1) {
						sb.append(MultiToken.DELIM_MULTI_TOKEN);
					}
				}

				if (j != ll.size() - 1) {
					sb.append("\t");
				}
			}
			if (i != ll.size() - 1) {
				sb.append("\n");
			}
		}

		return sb.toString().trim();
	}

	public void partition() throws Exception {
		DecimalFormat df = new DecimalFormat("000000");
		int batch_cnt = 0;

		List<String> lines = Generics.newArrayList();
		TextFileReader reader = new TextFileReader(PATPath.COL_FILE, "euc-kr");
		while (reader.hasNext()) {

			if (reader.getLineCnt() == 1) {
				continue;
			}

			String line = reader.next();

			if (lines.size() == 10000) {
				String outFileName = PATPath.COL_RAW_DIR + String.format("%s.txt.gz", df.format(batch_cnt++));
				FileUtils.writeStringCollection(outFileName, lines);
				lines.clear();
			}
			lines.add(line);
		}
		reader.close();

		if (lines.size() > 0) {
			String outFileName = PATPath.COL_RAW_DIR + String.format("%s.txt.gz", df.format(batch_cnt++));
			FileUtils.writeStringCollection(outFileName, lines);
		}
	}

	public void queryGloveModel() throws Exception {
		String dir = KPPath.DATA_DIR;

		GloveModel M = new GloveModel();
		M.readObject(dir + "glove_model_noun.ser.gz");

		LargeDocumentCollection lsc = new LargeDocumentCollection(dir + "col/noun_dc/");

		WordVectorModel vecs = new WordVectorModel(lsc.getVocab(), M.getAveragedModel());

		Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);

		WordVectorModel.interact(new WordSearcher(vecs, stopwords));
		System.out.println("process ends.");
	}

	public void tagPOS() throws Exception {
		Komoran komoran = new Komoran("lib/models-full/");

		for (File file : new File(PATPath.COL_LINE_DIR).listFiles()) {
			List<String> lines = Generics.newArrayList();

			for (String line : FileUtils.readLinesFromText(file)) {
				String[] parts = line.split("\t");
				parts = StrUtils.unwrap(parts);

				String content = parts[2];

				List<String> sents = Generics.newArrayList();

				for (String sent : content.split(StrUtils.LINE_REP)) {
					sent = getText(komoran.analyze(sent, 1));
					sents.add(sent.replace("\t", StrUtils.TAB_REP));
				}

				content = StrUtils.join(StrUtils.LINE_REP, sents);

				parts[2] = content;

				parts = StrUtils.wrap(parts);

				lines.add(StrUtils.join("\t", parts));
			}

			FileUtils.writeStringCollection(file.getPath().replace("line", "pos"), lines);
		}
	}

	public void tokenize() throws Exception {
		List<File> files = FileUtils.getFilesUnder(PATPath.COL_RAW_DIR);

		for (File file : files) {
			List<String> lines = Generics.newArrayList();

			for (String line : FileUtils.readLinesFromText(file)) {
				try {
					String[] parts = line.split("\t");
					parts = StrUtils.unwrap(parts);
					String content = parts[2];
					content = content.replace(".", ".\n");
					content = content.replace("\n", StrUtils.LINE_REP);
					parts[2] = content;
					parts = StrUtils.wrap(parts);
					lines.add(StrUtils.join("\t", parts));
				} catch (Exception e) {
					continue;
				}
			}

			FileUtils.writeStringCollection(file.getPath().replace("raw", "line"), lines);
		}
	}

}

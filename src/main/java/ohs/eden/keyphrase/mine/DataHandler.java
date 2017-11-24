package ohs.eden.keyphrase.mine;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.RawDocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.general.NLPUtils;
import ohs.matrix.DenseMatrix;
import ohs.ml.glove.CooccurrenceCounter;
import ohs.ml.glove.GloveModel;
import ohs.ml.glove.GloveParam;
import ohs.ml.glove.GloveTrainer;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.nlp.ling.types.LToken;
import ohs.tree.trie.hash.HMTrie;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListList;
import ohs.types.generic.Vocab;
import ohs.utils.ByteSize;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

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
		// dh.getSentences();
		// dh.tagPOS();
		// dh.trainGlove();
		// dh.matchPhrasesToKeywords();

		// dh.getWikiPhrases();
		// dh.getFrequentPhrases();

		// dh.getPaperKeywords();
		// dh.mergeKeywords();
		// dh.getMedicalPhrases();

		// dh.getPositiveData();

		// dh.getQualityTrainingPhrases();
		// dh.getMedicalTrainingPhrases();

		// dh.test();

		System.out.println("process ends.");
	}

	public void test() throws Exception {
		// RawDocumentCollection rdc = new RawDocumentCollection(KPPath.COL_DC_DIR);
		//
		// TextFileWriter writer = new TextFileWriter(KPPath.DATA_DIR + "kwd_doc.txt");
		// ListList<String> attrs = rdc.getAttrData();
		//
		// int doc_cnt1 = 0;
		// int doc_cnt2 = 0;
		// int kwd_cnt1 = 0;
		// int kwd_cnt2 = 0;
		//
		// CounterMap<String, String> cm = Generics.newCounterMap();
		//
		// for (int i = 0; i < rdc.size(); i++) {
		// HashMap<String, String> m = rdc.getMap(i);
		//
		// String kwdStr = m.get("kor_kwds");
		// String title = m.get("kor_title");
		// String abs = m.get("kor_abs");
		// String[] kwds = kwdStr.split(";");
		//
		// if (kwds.length == 0 || abs.length() == 0) {
		// continue;
		// }
		// // System.out.printf("%s\t%s\t%s\n", kwdStr, title, abs);
		//
		// doc_cnt1++;
		//
		// String content = title + "\n" + abs;
		//
		// Counter<String> c = Generics.newCounter();
		//
		// for (String kwd : kwds) {
		// kwd = kwd.trim();
		// if (kwd.length() == 0) {
		// continue;
		// }
		// kwd_cnt1++;
		// cm.incrementCount("KWD", kwd, 1);
		//
		// if (content.contains(kwd)) {
		// kwd_cnt2++;
		// cm.incrementCount("KWD_OCCUR", kwd, 1);
		// c.incrementCount(kwd, 1);
		// }
		// }
		//
		// if (c.size() > 0) {
		// doc_cnt2++;
		// writer.write(String.format("%d\t%s\t%s\t%s", i, kwdStr, title, abs));
		// writer.write("\n");
		// }
		// }
		// writer.close();
		//
		// System.out.println(doc_cnt1);
		// System.out.println(doc_cnt2);
		// System.out.println(kwd_cnt1);
		// System.out.println(kwd_cnt2);
	}

	public void get3PKeywords() throws Exception {
		Counter<String> c = Generics.newCounter();

		for (File file : FileUtils.getFilesUnder(KPPath.COL_LINE_POS_DIR)) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] parts = line.split("\t");
				parts = StrUtils.unwrap(parts);

				String kwdStr1 = parts[2];
				String kwdStr2 = parts[8];

				String[] kwds1 = kwdStr1.split(";");
				String[] kwds2 = kwdStr2.split(";");

				if (kwds1.length == 0 || kwds1.length != kwds2.length) {
					continue;
				}

				for (int i = 0; i < kwds1.length; i++) {
					String kwd1 = kwds1[i];
					String kwd2 = kwds2[i];

					if (kwd1.length() == 0 || kwd2.length() == 0) {
						continue;
					}
					kwd2 = kwd2.replace(" / ", "/").replace(" + ", " ").replace("<nl>", " ").replace(" ", "_");
					c.incrementCount(kwd2, 1);
				}
			}
		}

		double avg_len = 0;

		int max_len = 0;

		for (String kwd : c.keySet()) {
			avg_len += kwd.split(" ").length * c.getCount(kwd);
			max_len = Math.max(max_len, kwd.split(" ").length);
		}

		avg_len /= c.totalCount();

		System.out.printf("avg_len: %f\n", avg_len);
		System.out.printf("max_len: %d\n", max_len);

		FileUtils.writeStringCounterAsText(KPPath.KP_DIR + "phrs_3p_kwds.txt.gz", c);
	}

	public void get3PKeywordsEng() throws Exception {
		RawDocumentCollection rdc = new RawDocumentCollection(KPPath.COL_DC_DIR);

		Counter<String> c = Generics.newCounter();

		for (int i = 0; i < rdc.size(); i++) {
			int progress = BatchUtils.progress(i + 1, rdc.size());
			if (progress > 0) {
				System.out.printf("[%d]\n", progress);
			}
			HashMap<String, String> m = rdc.getMap(i);
			List<String> vals = rdc.get(i);
			int j = 0;
			String type = vals.get(j++);
			String cn = vals.get(j++);
			String korKwsStr = vals.get(j++);
			String engKwsStr = vals.get(j++);

			if (type.equals("patent")) {
				break;
			}

			for (String kwd : engKwsStr.split(";")) {
				if (kwd.length() > 0) {
					c.incrementCount(kwd, 1);
				}
			}
		}

		FileUtils.writeStringCounterAsText(KPPath.KP_DIR + "eng_kwds.txt", c);

		System.out.println(c);
	}

	public void getFrequentPhrases() throws Exception {
		Timer timer = Timer.newTimer();

		// String dataDir = MIRPath.TREC_CDS_2016_DIR;
		String dataDir = "../../data/naver_news/";

		PhraseCollection pc = new PhraseCollection(dataDir + "col/dc/");
		Counter<String> c = Generics.newCounter();

		int[][] ranges = BatchUtils.getBatchRanges(pc.size(), 100);
		long len = 0;

		for (int i = 0, j = 0; i < ranges.length; i++) {
			int[] range = ranges[i];

			List<SPostingList> pls = pc.getPostingLists(range[0], range[1]);
			for (SPostingList pl : pls) {
				String phrs = pl.getPhrase();
				c.incrementCount(phrs, pl.getPosData().sizeOfEntries());

				len += phrs.length() + Double.BYTES;

				j++;
				int prog = BatchUtils.progress(j, pc.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s, %s]\n", prog, j, pc.size(), new ByteSize(len).toString(),
							timer.stop());
				}
			}
		}

		FileUtils.writeStringCounterAsText(dataDir + "phrs/phrs_freq.txt", c);
	}

	public void getMedicalTrainingPhrases() throws Exception {

		String dir = MIRPath.TREC_CDS_2016_DIR;

		Counter<String> cdsPhrss = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(dir + "phrs/phrs_freq.txt")) {
			String[] parts = line.split("\t");
			cdsPhrss.incrementCount(parts[0].toLowerCase(), Double.parseDouble(parts[1]));
		}

		Counter<String> medPhrss = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(dir + "phrs/phrs_m_medical.txt")) {
			String[] parts = line.split("\t");
			medPhrss.setCount(parts[0].toLowerCase().replace("_", " "), 1);
		}

		Counter<String> goodPhrss = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(dir + "phrs/phrs_q_good.txt")) {
			String[] parts = line.split("\t");
			goodPhrss.setCount(parts[0].toLowerCase().replace("_", " "), 1);
		}

		Counter<String> notMedPhrss = Generics.newCounter();

		for (String phrs : cdsPhrss.getSortedKeys()) {
			double cnt = cdsPhrss.getCount(phrs);
			phrs = phrs.replace("_", " ");

			String label = "not_medical";

			if (goodPhrss.containsKey(phrs)) {
				if (!medPhrss.containsKey(phrs)) {
					notMedPhrss.setCount(phrs, cnt);
				}
			}
		}

		FileUtils.writeStringCounterAsText(dir + "phrs/phrs_m_not_medical.txt", notMedPhrss);
	}

	public void getPaperKeywords() throws Exception {
		String[] dirs = { MIRPath.TREC_CDS_2014_COL_DC_DIR, MIRPath.TREC_CDS_2016_COL_DC_DIR,
				MIRPath.DATA_DIR + "scopus/col/dc/" };

		for (int i = 0; i < dirs.length; i++) {
			String dir = dirs[i];
			// if (i != 2) {
			// continue;
			// }

			RawDocumentCollection rdc = new RawDocumentCollection(dir);
			System.out.printf("[%s]\n", dir);
			System.out.println(rdc.size());
			System.out.println(rdc.getAttrData());
			Counter<String> c = Generics.newCounter();

			int[][] ranges = BatchUtils.getBatchRanges(rdc.size(), 1000);

			for (int j = 0; j < ranges.length; j++) {
				int[] range = ranges[j];
				ListList<String> res = rdc.getRange(range);

				for (int k = 0; k < res.size(); k++) {
					List<String> l = res.get(k);
					Map<String, String> m = rdc.getMap(l);
					String kwds = m.get("kwds");

					if (kwds != null && kwds.length() > 0) {
						for (String kwd : kwds.split("\n")) {
							if (!kwd.endsWith(".")) {
								c.incrementCount(kwd.toLowerCase(), 1);
							}
						}
					}
				}
			}

			// EnglishNormalizer sn = new EnglishNormalizer(true);

			Counter<String> c2 = Generics.newCounter(c.size());

			for (Entry<String, Double> e : c.entrySet()) {
				String kwd = e.getKey();
				double cnt = e.getValue();
				// kwd = sn.normalize(kwd);
				c2.setCount(kwd, cnt);
			}

			FileUtils.writeStringCounterAsText(dir.replace("col/dc/", "phrs/kwds.txt"), c2);
		}
	}

	public void getPositiveData() throws Exception {

		{
			List<String> ms = Generics.newArrayList();
			List<String> qs = Generics.newArrayList();

			int size = 0;
			TextFileReader reader = new TextFileReader(MIRPath.TREC_CDS_2016_DIR + "phrs/cpts.txt");
			while (reader.hasNext()) {
				String[] ps = reader.next().split("\t");
				if (reader.getLineCnt() == 1) {
					size = Integer.parseInt(ps[1]);
					ms = Generics.newArrayList(size);
					qs = Generics.newArrayList(size);
					continue;
				}

				String cpt = ps[0];

				Set<String> names = Generics.newHashSet();

				for (int i = 2; i < ps.length; i++) {
					names.add(ps[i]);
				}

				boolean is_in_mesh = names.contains("mes");
				boolean is_in_snomed_ct = names.contains("sno");
				boolean is_in_trec_cds = names.contains("cds");
				boolean is_in_wiki = names.contains("wkt");
				boolean is_in_scopus = names.contains("sco");

				if (is_in_wiki && (is_in_scopus || is_in_trec_cds)) {
					qs.add(cpt);

					if (is_in_mesh || is_in_snomed_ct) {
						ms.add(cpt);
					}
				} else {
				}
			}
			reader.close();

			FileUtils.writeStringCollectionAsText(MIRPath.TREC_CDS_2016_DIR + "phrs/phrs_q_good.txt", qs);
			FileUtils.writeStringCollectionAsText(MIRPath.TREC_CDS_2016_DIR + "phrs/phrs_m_medical.txt", ms);
		}
	}

	public void getQualityTrainingPhrases() throws Exception {
		String dir = MIRPath.TREC_CDS_2016_DIR;

		Counter<String> cdsPhrss = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(dir + "phrs/phrs_freq.txt")) {
			String[] parts = line.split("\t");
			cdsPhrss.incrementCount(parts[0].toLowerCase().replace("_", " "), Double.parseDouble(parts[1]));
		}

		Counter<String> goodPhrss = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(dir + "phrs/phrs_q_good.txt")) {
			String[] parts = line.split("\t");
			goodPhrss.setCount(parts[0].toLowerCase().replace("_", " "), 1);
		}

		String[] dirs = { MIRPath.DATA_DIR + "wiki/phrs_title.txt", MIRPath.DATA_DIR + "wiki/phrs_link.txt" };

		Counter<String> wikiPhrss = Generics.newCounter();

		for (int i = 0; i < dirs.length; i++) {
			String d = dirs[i];
			Counter<String> c = FileUtils.readStringCounterFromText(d);

			for (String phrs : c.keySet()) {
				if (phrs.length() > 1) {
					wikiPhrss.incrementCount(phrs, 1);
				}
			}
		}

		Counter<String> badPhrss = Generics.newCounter();
		Counter<String> notBadPhrss = Generics.newCounter();

		for (String phrs : cdsPhrss.getSortedKeys()) {
			double cnt = cdsPhrss.getCount(phrs);
			if (!goodPhrss.containsKey(phrs)) {
				if (wikiPhrss.containsKey(phrs)) {
					notBadPhrss.setCount(phrs, cnt);
				} else {
					badPhrss.setCount(phrs, cnt);
				}
			}

		}
		FileUtils.writeStringCounterAsText(dir + "phrs/phrs_q_bad.txt", badPhrss);
		FileUtils.writeStringCounterAsText(dir + "phrs/phrs_q_not_bad.txt", notBadPhrss);
	}

	public void getSentences() throws Exception {
		String[] dirs = { MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.DATA_DIR + "scopus/col/dc/" };

		for (int i = 0; i < dirs.length; i++) {
			String dir = dirs[i];
			RawDocumentCollection rdc = new RawDocumentCollection(dir);

			for (int j = 0; j < rdc.size(); j++) {
				Map<String, String> avm = rdc.getMap(j);

				String title = avm.get("title");
				String abs = avm.get("abs");

				String kwdStr = avm.get("kwds");

				Set<String> kwdSet = Generics.newHashSet();

				for (String kwd : kwdStr.split("\n")) {
					kwdSet.add(kwd.toLowerCase());
				}

				List<String> sents = Generics.newArrayList();
				sents.add(title.toLowerCase());

				for (String sent : abs.split("\n")) {
					sents.add(sent.toLowerCase());
				}

				for (String sent : sents) {
					List<String> words = StrUtils.split(sent);

					HMTrie<String> dict = PhraseMapper.createTrie(kwdSet);
					PhraseMapper m = new PhraseMapper(dict);

					List<ohs.types.generic.Pair<Integer, Integer>> res = m.map(words);

					if (res.size() > 0) {
						System.out.println();
					}
				}

			}
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

					sb.append(String.format("%s%s%s", f, LToken.DELIM, s));
					// sb.append(String.format("%s%s%s", f, "/", s));

					if (k != l.size() - 1) {
						// sb.append("+");
						sb.append(" ");
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

	public void getWikiPhrases() throws Exception {
		Timer timer = Timer.newTimer();
		RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.WIKI_COL_DC_DIR);

		int[][] ranges = BatchUtils.getBatchRanges(rdc.size(), 1000);

		Counter<String> c1 = Generics.newCounter();
		Counter<String> c2 = Generics.newCounter();

		String reg = "\\([^\\(\\)]+\\)";

		Pattern p = Pattern.compile(reg);

		for (int i = 0; i < ranges.length; i++) {
			int[] range = ranges[i];

			ListList<String> ps = rdc.getValues(range);

			for (int j = 0; j < ps.size(); j++) {
				List<String> vals = ps.get(j);

				String title = vals.get(2);
				String phrss = vals.get(4);

				Matcher m = p.matcher(title);
				StringBuffer sb = new StringBuffer();

				if (m.find()) {
					String g = m.group();
					// System.out.println(g);
					// int s = m.start();
					m.appendReplacement(sb, "");
				}
				m.appendTail(sb);

				title = sb.toString().trim();

				if (title.startsWith("List of")) {
					title = title.substring("List of".length() + 1);
				}

				String t = title;

				// t = sn.normalize(t);

				if (t.split(" ").length > 1 && !t.contains("?")) {
					c1.incrementCount(t, 1);
				}

				for (String phrs : phrss.split("\\|")) {
					// phrs = StrUtils.join(" ", NLPUtils.tokenize(title));
					// phrs = sn.normalize(phrs);
					if (phrs.split(" ").length > 1 && !phrs.contains("?")) {
						c2.incrementCount(phrs, 1);
					}
				}
			}

			int prog = BatchUtils.progress(i + 1, ranges.length);

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d, %s]\n", prog, i + 1, ranges.length, timer.stop());
			}
		}

		FileUtils.writeStringCounterAsText(MIRPath.WIKI_DIR + "phrs_title.txt", c1);
		FileUtils.writeStringCounterAsText(MIRPath.WIKI_DIR + "phrs_link.txt", c2);
	}

	public void matchPhrasesToKeywords() throws Exception {
		Counter<String> c = FileUtils.readStringCounterFromText(KPPath.KP_DIR + "phrs_3p_kwds.txt.gz");

		List<String> lines = FileUtils.readLinesFromText(KPPath.KP_DIR + "phrs_3p.txt.gz");

		Counter<String> cc = Generics.newCounter();

		for (int i = 0; i < lines.size(); i++) {
			List<String> parts = Generics.newArrayList();

			for (String s : lines.get(i).split("\t")) {
				parts.add(s);
			}

			int cnt = (int) c.getCount(parts.get(0));

			if (cnt > 0) {
				cc.incrementCount("Phrs+Kwd", 1);
			}

			cc.incrementCount("Phrs", 1);

			parts.add(cnt + "");

			lines.set(i, StrUtils.join("\t", parts));
		}

		System.out.println(cc);

		FileUtils.writeStringCollectionAsText(KPPath.KP_DIR + "phrs_3p_label.txt.gz", lines);
	}

	public void mergeKeywords() throws Exception {
		String[] dirs = { MIRPath.DATA_DIR + "trec_cds/2016/phrs/kwds.txt", MIRPath.DATA_DIR + "scopus/phrs/kwds.txt",
				MIRPath.DATA_DIR + "mesh/phrss.txt", MIRPath.DATA_DIR + "snomed_ct/phrss.txt",
				MIRPath.DATA_DIR + "wiki/phrs_title.txt", MIRPath.DATA_DIR + "wiki/phrs_link.txt" };
		String[] names = { "cds", "sco", "mes", "sno", "wkt", "wkt" };

		CounterMap<String, String> cm = Generics.newCounterMap(2000000);

		{
			for (int i = 0; i < dirs.length; i++) {
				String dir = dirs[i];
				String name = names[i];
				Counter<String> c = FileUtils.readStringCounterFromText(dir);

				for (String phrs : c.keySet()) {
					double cnt = c.getCount(phrs);
					// if (phrs.length() > 1) {
					cm.incrementCount(phrs, name, cnt);
					// }
				}
			}

			CounterMap<String, String> cm2 = cm.invert();
			Counter<String> c = Generics.newCounter();

			for (String src : cm2.keySet()) {
				c.setCount(src, cm2.getCounter(src).size());
			}

			System.out.println(c.toStringSortedByValues(true, true, c.size(), "\t"));
		}

		Counter<String> c = Generics.newCounter(cm.size());

		for (Entry<String, Counter<String>> e : cm.getEntrySet()) {
			Counter<String> c2 = e.getValue();
			if (c2.size() > 1) {
				c.setCount(e.getKey(), c2.size());
			}
		}

		List<String> res = Generics.newArrayList(cm.size());

		for (String phrs : c.getSortedKeys()) {
			int len = (int) c.getCount(phrs);
			Counter<String> c2 = cm.removeKey(phrs);
			List<String> rs = Generics.newArrayList(c2.keySet());
			Collections.sort(rs);
			res.add(phrs + "\t" + len + "\t" + StrUtils.join("\t", rs));
		}

		// FileUtils.writeStringCounterMapAsText("../../data/medical_ir/trec_cds/2016/phrs/cpts.txt",
		// cm);
		// FileUtils.writeStringCollectionAsText("../../data/medical_ir/trec_cds/2016/phrs/cpts.txt",
		// res);
	}

	public void getMedicalPhrases() throws Exception {
		List<String> lines = FileUtils.readLinesFromText("../../data/medical_ir/phrs/phrs.txt");
		List<String> phrss = Generics.newArrayList();

		for (String line : lines) {
			String[] ps = line.split("\t");
			String phrs = ps[0];
			Set<String> srcs = Generics.newHashSet();

			for (int i = 2; i < ps.length; i++) {
				srcs.add(ps[i]);
			}

			boolean in_mesh = srcs.contains("mes") ? true : false;
			boolean in_snomed = srcs.contains("sno") ? true : false;
			boolean in_cds = srcs.contains("cds") ? true : false;

			if (in_mesh && in_snomed) {
				phrss.add(phrs);
			}
		}

		FileUtils.writeStringCollectionAsText("../../data/medical_ir/phrs/phrs_medical.txt", phrss);

	}

	public String normalize(String s) {
		s = StrUtils.join(" ", NLPUtils.tokenize(s));
		s = StrUtils.normalizeNumbers(s);
		s = StrUtils.normalizeSpaces(s);
		return s;
	}

	public void queryGloveModel() throws Exception {
		String dir = KPPath.DATA_DIR;
		//
		// GloveModel M = new GloveModel();
		// M.readObject(dir + "glove_model_noun.ser.gz");
		//
		// DocumentCollection lsc = new DocumentCollection(dir +
		// "col/noun_dc/");
		//
		// WordVectorModel vecs = new WordVectorModel(lsc.getVocab(),
		// M.getAveragedModel());
		//
		// Set<String> stopwords =
		// FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		//
		// WordSearcher.interact(new WordSearcher(vecs, stopwords));
		System.out.println("process ends.");
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

	public void trainGlove() throws Exception {
		String dir = KPPath.COL_DIR;
		String scDir = dir + "dc/";
		String ccDir = dir + "cocnt/";
		String outFileName1 = KPPath.KP_DIR + "glove_model.ser.gz";
		String outFileName2 = KPPath.KP_DIR + "glove_embedding.ser.gz";

		int thread_size = 50;
		int hidden_size = 200;
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
	}

}

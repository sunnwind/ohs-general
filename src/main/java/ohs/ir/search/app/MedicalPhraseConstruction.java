package ohs.ir.search.app;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ohs.corpus.type.RawDocumentCollection;
import ohs.corpus.type.SimpleStringNormalizer;
import ohs.corpus.type.StringNormalizer;
import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListList;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class MedicalPhraseConstruction {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		MedicalPhraseConstruction b = new MedicalPhraseConstruction();
		// b.getPaperKeywords();
		// b.getWikiPhrases();
		// b.mergePhrases();
		// b.getMedicalPhrases();

		System.out.println("process ends.");
	}

	public void getPaperKeywords() throws Exception {
		String[] dirs = { MIRPath.TREC_CDS_2014_COL_DC_DIR, MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.DATA_DIR + "scopus/col/dc/" };

		for (int i = 0; i < dirs.length; i++) {
			String dir = dirs[i];

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

			SimpleStringNormalizer sn = new SimpleStringNormalizer(true);

			Counter<String> c2 = Generics.newCounter(c.size());

			for (Entry<String, Double> e : c.entrySet()) {
				String kwd = e.getKey();
				double cnt = e.getValue();
				kwd = sn.normalize(kwd);
				c2.setCount(kwd, cnt);
			}
			FileUtils.writeStringCounterAsText(dir.replace("col/dc/", "phrs/kwds.txt"), c2);
		}
	}

	public void getMedicalPhrases() throws Exception {
		List<String> phrss = Generics.newArrayList();

		int size = 0;
		TextFileReader reader = new TextFileReader(MIRPath.DATA_DIR + "phrs.txt");
		while (reader.hasNext()) {
			String[] ps = reader.next().split("\t");
			if (reader.getLineCnt() == 1) {
				size = Integer.parseInt(ps[1]);
				phrss = Generics.newArrayList(size);
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

			if (is_in_wiki && (is_in_scopus || is_in_trec_cds) && (is_in_mesh || is_in_snomed_ct)) {
				phrss.add(cpt);
			}
		}
		reader.close();

		Counter<String> c = Generics.newCounter();

		for (String phrs : phrss) {
			for (String word : phrs.split(" ")) {
				c.incrementCount(word, 1);
			}
		}

		FileUtils.writeStringCollectionAsText(MIRPath.DATA_DIR + "phrs_medical.txt", phrss);
		FileUtils.writeStringCounterAsText(MIRPath.DATA_DIR + "word_medical.txt", c);
	}

	public void getWikiPhrases() throws Exception {
		Timer timer = Timer.newTimer();
		RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.WIKI_COL_DC_DIR);

		int[][] ranges = BatchUtils.getBatchRanges(rdc.size(), 1000);

		Counter<String> c1 = Generics.newCounter();
		Counter<String> c2 = Generics.newCounter();

		String reg = "\\([^\\(\\)]+\\)";

		Pattern p = Pattern.compile(reg);

		StringNormalizer sn = new SimpleStringNormalizer(true);

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

				t = sn.normalize(t);

				if (t.split(" ").length > 1 && !t.contains("?")) {
					c1.incrementCount(t, 1);
				}

				for (String phrs : phrss.split("\\|")) {
					// phrs = StrUtils.join(" ", NLPUtils.tokenize(title));
					phrs = sn.normalize(phrs);
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

	public void mergePhrases() throws Exception {
		String[] dirs = { MIRPath.DATA_DIR + "trec_cds/2016/phrs/kwds.txt", MIRPath.DATA_DIR + "scopus/phrs/kwds.txt",
				MIRPath.DATA_DIR + "mesh/phrss.txt", MIRPath.DATA_DIR + "snomed_ct/phrss.txt", MIRPath.DATA_DIR + "wiki/phrs_title.txt",
				MIRPath.DATA_DIR + "wiki/phrs_link.txt" };
		String[] names = { "cds", "sco", "mes", "sno", "wkt", "wkt" };

		CounterMap<String, String> cm = Generics.newCounterMap(2000000);

		for (int i = 0; i < dirs.length; i++) {
			String dir = dirs[i];
			String name = names[i];
			Counter<String> c = FileUtils.readStringCounterFromText(dir);

			for (String phrs : c.keySet()) {
				double cnt = c.getCount(phrs);
				if (phrs.length() > 1) {
					cm.incrementCount(phrs, name, cnt);
				}
			}
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

		FileUtils.writeStringCollectionAsText("../../data/medical_ir/phrss.txt", res);

	}

}

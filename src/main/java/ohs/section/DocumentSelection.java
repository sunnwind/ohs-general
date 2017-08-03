package ohs.section;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.ir.weight.TermWeighting;
import ohs.types.generic.Counter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.UnicodeUtils;

public class DocumentSelection {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DocumentSelection dh = new DocumentSelection();
		dh.test();

		System.out.println("process ends.");
	}

	public void step1() {

	}

	private List<File> files;

	private String regex = "[\\s\\p{Punct}]+";

	public DocumentSelection() {
		files = FileUtils.getFilesUnder(KPPath.COL_LINE_DIR + "paper");
	}

	public void test() throws Exception {

		Counter<String> docFreqs = Generics.newCounter(5000000);
		Set<String> selected = Generics.newHashSet();

		double num_docs = 0;

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			List<String> lines = FileUtils.readLinesFromText(file);

			num_docs += lines.size();

			for (int j = 0; j < lines.size(); j++) {
				String[] ps = lines.get(j).split("\t");
				ps = StrUtils.unwrap(ps);

				int k = 0;

				String type = ps[k++];
				String cn = ps[k++];
				String korKwdStr = ps[k++];
				String engKwdStr = ps[k++];
				String korTitle = ps[k++];
				String engTitle = ps[k++];
				String korAbs = ps[k++];
				String engAbs = ps[k++];

				if (korTitle.length() == 0 || korAbs.length() == 0 || korKwdStr.length() == 0) {
					continue;
				}

				List<String> kwds = StrUtils.split(StrUtils.LINE_REP, korKwdStr);

				// System.out.println(korTitle + "\n" + korAbs);
				// System.out.println();

				String content = korTitle + "\n" + korAbs;

				int cnt = 0;
				for (String kwd : kwds) {
					if (content.contains(kwd)) {
						cnt++;
					}
				}

				if (cnt == 0) {
					continue;
				}

				selected.add(cn);

				Counter<String> c = Generics.newCounter();

				for (String word : StrUtils.split(regex, content)) {
					word = word.toLowerCase();
					if (UnicodeUtils.isKorean(word)) {
						c.incrementCount(word, 1);
					} else {
						// System.out.println(word);
					}
				}

				docFreqs.incrementAll(c.keySet(), 1);
			}
		}

		Counter<String> docWeights = Generics.newCounter(selected.size());

		Map<String, String> cnToJournal = Generics.newHashMap();

		for (String line : FileUtils.readLinesFromText(KPPath.KP_DIR + "170801_paper_cn_jtk.txt")) {
			String[] ps = line.split("\t");
			if (ps.length == 2) {
				cnToJournal.put(ps[0], ps[1]);
			}
		}

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			List<String> lines = FileUtils.readLinesFromText(file);

			for (int j = 0; j < lines.size(); j++) {
				String[] ps = lines.get(j).split("\t");
				ps = StrUtils.unwrap(ps);

				int k = 0;

				String type = ps[k++];
				String cn = ps[k++];
				String korKwdStr = ps[k++];
				String engKwdStr = ps[k++];
				String korTitle = ps[k++];
				String engTitle = ps[k++];
				String korAbs = ps[k++];
				String engAbs = ps[k++];

				if (!selected.contains(cn) || cnToJournal.get(cn) == null) {
					continue;
				}

				List<String> kwds = StrUtils.split(StrUtils.LINE_REP, korKwdStr);

				// System.out.println(korTitle + "\n" + korAbs);
				// System.out.println();

				String content = korTitle + "\n" + korAbs;

				Counter<String> c = Generics.newCounter();

				for (String word : StrUtils.split(regex, content)) {
					word = word.toLowerCase();
					double doc_freq = docFreqs.getCount(word);
					if (UnicodeUtils.isKorean(word) || doc_freq < 5) {
						c.incrementCount(word.toLowerCase(), 1);
					}
				}

				for (String word : c.keySet()) {
					double cnt = c.getCount(word);
					double doc_freq = docFreqs.getCount(word);
					double tfidf = TermWeighting.tfidf(cnt, num_docs, doc_freq);
					c.setCount(word, tfidf);
				}

				if (c.size() == 0) {
					continue;
				}

				double sum = 1d * c.totalCount() / c.size();
				docWeights.setCount(cn, sum);
			}
		}

		docWeights.keepTopNKeys(30000);

		Map<String, Integer> ranks = Generics.newHashMap();

		{
			int rank = 0;
			for (String cn : docWeights.getSortedKeys()) {
				ranks.put(cn, ++rank);
			}
		}

		FileUtils.deleteFilesUnder(KPPath.KP_DIR + "subset");

		DecimalFormat df = new DecimalFormat("000000");

		Map<String, String> meta = Generics.newHashMap();

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			List<String> lines = FileUtils.readLinesFromText(file);

			for (int j = 0; j < lines.size(); j++) {
				String[] ps = lines.get(j).split("\t");
				ps = StrUtils.unwrap(ps);

				int k = 0;

				String type = ps[k++];
				String cn = ps[k++];
				String korKwdStr = ps[k++];
				String engKwdStr = ps[k++];
				String korTitle = ps[k++];
				String engTitle = ps[k++];
				String korAbs = ps[k++];
				String engAbs = ps[k++];
				String journal = cnToJournal.get(cn);

				if (!docWeights.containsKey(cn)) {
					continue;
				}

				String content = korTitle + "\n\n" + korAbs + "\n";
				int rank = ranks.get(cn);

				String outFileName = String.format("%s/%s/%s_%s.txt", KPPath.KP_DIR, "subset", df.format(rank), cn);
				FileUtils.writeAsText(outFileName, content);

				List<String> items = Generics.newArrayList();
				items.add(cn);
				items.add(journal);

				for (String kp : korKwdStr.split(StrUtils.LINE_REP)) {
					items.add(kp);
				}

				meta.put(df.format(rank), StrUtils.join("\t", items));
			}
		}

		for (File file : FileUtils.getFilesUnder(KPPath.KP_DIR + "subset")) {
			String fileName = file.getName();
			String[] ps = fileName.split("[_\\.]+");
			fileName = String.format("%s.%s", ps[0], ps[2]);
			file.renameTo(new File(file.getParentFile(), fileName));
		}

		String outFileName = String.format("%s/%s/meta.txt", KPPath.KP_DIR, "subset");

		List<String> res = Generics.newArrayList();

		for (String key : Generics.newTreeSet(meta.keySet())) {
			res.add(key + "\t" + meta.get(key));
		}

		FileUtils.writeAsText(outFileName, StrUtils.join("\n", res));

	}

}

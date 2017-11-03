package ohs.eden.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.bitbucket.eunjeon.seunjeon.Analyzer;
import org.bitbucket.eunjeon.seunjeon.LNode;
import org.bitbucket.eunjeon.seunjeon.Morpheme;

import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.ir.weight.TermWeighting;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.CounterMapMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import scala.collection.mutable.WrappedArray;

public class DataHandler {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DataHandler dh = new DataHandler();

		// dh.index();
		// dh.search();

		// dh.extractExcelFiles();
		// dh.extractNtisText();
		// dh.tokenize();

		// dh.extractKeywords();
		// dh.computeKeywordWeights();
		// dh.mapKeywords();
		// dh.getCocounts();
		dh.getCocounts2();
		// dh.getYearCocounts();
		// dh.format();

		System.out.println("process ends.");
	}

	public void extractKeywords() throws Exception {
		String[] dirNames = { "paper", "ntis" };

		Counter<String> c = Generics.newCounter();

		for (String dirName : dirNames) {
			for (File file : FileUtils.getFilesUnder(KPPath.COL_DIR + "line_2/" + dirName)) {
				List<String> lines = FileUtils.readLinesFromText(file);
				for (String line : lines) {
					List<String> ps = StrUtils.split("\t", line);

					if (ps.size() != 9) {
						continue;
					}

					ps = StrUtils.unwrap(ps);

					String kwdStr = ps.get(2);

					if (kwdStr.length() == 0) {
						continue;
					}

					for (String kwd : kwdStr.split("<nl>")) {
						kwd = kwd.replace(".", "");
						kwd = StrUtils.normalizeSpaces(kwd);

						try {
							kwd = StrUtils.normalizePunctuations(kwd);
						} catch (Exception e) {
							continue;
						}

						if (kwd.matches("^\\p{Punct}+$") || kwd.matches("^\\d+$")) {
							continue;
						}

						if (kwd.length() > 1) {
							c.incrementCount(kwd, 1);
						}
					}
				}
			}
		}

		FileUtils.writeStringCounterAsText(KPPath.DATA_DIR + "scenario/kwds.txt", c);
	}

	public void computeKeywordWeights() throws Exception {
		Counter<String> c = FileUtils.readStringCounterFromText(KPPath.DATA_DIR + "scenario/kwds.txt");

		CandidatePhraseSearcher cps = CandidatePhraseSearcher.newCandidatePhraseSearcher(c.keySet());

		CounterMap<String, String> kwdTypeDocFreqs = Generics.newCounterMap();
		CounterMap<String, String> kwdTypeCnts = Generics.newCounterMap();

		Counter<String> numDocs = Generics.newCounter();

		for (File file : FileUtils.getFilesUnder(KPPath.COL_DIR + "line_2/")) {
			for (String line : FileUtils.readLinesFromText(file)) {
				List<String> ps = StrUtils.split("\t", line);

				if (ps.size() != 9) {
					continue;
				}

				ps = StrUtils.unwrap(ps);

				String type = ps.get(0);
				String title = ps.get(4);
				String abs = ps.get(5);
				String date = ps.get(8).trim();

				if (date.length() == 8) {

				} else if (date.length() == 6) {
					date = date + "00";
				} else if (date.length() == 4) {
					date = date + "0000";
				} else {
					date = date.replace("-", "");
				}

				if (date.length() == 0 || !date.matches("^\\d+$") || date.equals("0000")) {
					continue;
				}

				String content = title + "<nl>" + abs;

				MDocument d = new MDocument();

				for (String sent : content.split("<nl>")) {
					MSentence s = new MSentence();

					for (String word : StrUtils.split(sent)) {
						MToken t = new MToken();
						t.add(word);
						s.add(t);
					}
					d.add(s);
				}

				List<List<IntPair>> locData = cps.search(d);

				Counter<String> cc = Generics.newCounter();

				for (int i = 0; i < locData.size(); i++) {
					MSentence s = d.get(i);
					List<IntPair> locs = locData.get(i);

					for (IntPair p : locs) {
						String kwd = StrUtils.join(" ", s.subSentence(p.getFirst(), p.getSecond()).getTokenStrings(0));
						cc.incrementCount(kwd, 1);
					}
				}

				for (String kwd : cc.keySet()) {
					kwdTypeCnts.incrementCount(kwd, type, cc.getCount(kwd));
					kwdTypeDocFreqs.incrementCount(kwd, type, 1);
				}
				numDocs.incrementCount(type, 1);
			}
		}

		String[] colTypes = { "paper", "patent", "ntis" };

		CounterMap<String, String> cm4 = Generics.newCounterMap();

		for (String kwd : kwdTypeDocFreqs.keySet()) {
			Counter<String> cnts = kwdTypeCnts.getCounter(kwd);
			Counter<String> docFreqs = kwdTypeDocFreqs.getCounter(kwd);

			for (String type : colTypes) {
				double num_docs = numDocs.getCount(type) + 1;
				double doc_freq = docFreqs.getCount(type) + 1;
				double cnt = cnts.getCount(type) + 1;
				double tfidf = TermWeighting.tfidf(cnt, num_docs, doc_freq);
				cm4.setCount(kwd, type, tfidf);
			}
		}

		FileUtils.writeStringCounterMapAsText(KPPath.DATA_DIR + "scenario/kwds_cnt.txt", kwdTypeCnts);
		FileUtils.writeStringCounterMapAsText(KPPath.DATA_DIR + "scenario/kwds_tfidf.txt", cm4);
	}

	public void mapKeywords() throws Exception {

		CandidatePhraseSearcher cps = null;

		{
			Counter<String> c = FileUtils.readStringCounterFromText(KPPath.DATA_DIR + "scenario/kwds.txt");
			cps = CandidatePhraseSearcher.newCandidatePhraseSearcher(c.keySet());
		}

		List<String> res = Generics.newLinkedList();

		for (File file : FileUtils.getFilesUnder(KPPath.COL_DIR + "line_2/")) {
			for (String line : FileUtils.readLinesFromText(file)) {
				List<String> ps = StrUtils.split("\t", line);

				if (ps.size() != 9) {
					continue;
				}

				ps = StrUtils.unwrap(ps);

				String type = ps.get(0);
				String title = ps.get(4);
				String abs = ps.get(5);
				String date = ps.get(8).trim();

				if (date.length() == 8) {

				} else if (date.length() == 6) {
					date = date + "00";
				} else if (date.length() == 4) {
					date = date + "0000";
				} else {
					date = date.replace("-", "");
				}

				if (date.length() < 4 || !date.matches("^\\d+$") || date.equals("0000")) {
					continue;
				}

				if (type.equals("report")) {
					continue;
				}

				String content = title + "<nl>" + abs;

				MDocument d = new MDocument();

				for (String sent : content.split("<nl>")) {
					MSentence s = new MSentence();

					for (String word : StrUtils.split(sent)) {
						MToken t = new MToken();
						t.add(word);
						s.add(t);
					}
					d.add(s);
				}

				List<List<IntPair>> locData = cps.search(d);

				Counter<String> tmp = Generics.newCounter();

				for (int i = 0; i < locData.size(); i++) {
					MSentence s = d.get(i);
					List<IntPair> locs = locData.get(i);

					for (IntPair p : locs) {
						String kwd = StrUtils.join(" ", s.subSentence(p.getFirst(), p.getSecond()).getTokenStrings(0));
						tmp.incrementCount(kwd, 1);
					}
				}

				if (tmp.size() == 0) {
					continue;
				}

				String year = date.substring(0, 4);

				List<String> l = Generics.newLinkedList();
				l.add(type);
				l.add(year);

				for (String kwd : tmp.getSortedKeys()) {
					int cnt = (int) tmp.getCount(kwd);
					l.add(String.format("%s:%d", kwd, cnt));
				}

				res.add(StrUtils.join("\t", l));
			}
		}

		FileUtils.writeStringCollectionAsText(KPPath.DATA_DIR + "scenario/doc_kwds.txt", res);
	}

	public void getCocounts() throws Exception {
		Set<String> targets = Generics
				.newHashSet(FileUtils.readLinesFromText(KPPath.DATA_DIR + "scenario/kwds_target.txt"));

		CounterMapMap<String, String, String> cmm = Generics.newCounterMapMap();

		for (String line : FileUtils.readLinesFromText(KPPath.DATA_DIR + "scenario/doc_kwds.txt")) {
			String[] ps = line.split("\t");
			String type = ps[0];
			String year = ps[1];
			Counter<String> c = Generics.newCounter(ps.length - 2);

			for (int i = 2; i < ps.length; i++) {
				String p = ps[i];
				String[] two = StrUtils.split2Two(":", p);
				c.incrementCount(two[0], Double.parseDouble(two[1]));
			}

			List<String> kwds = c.getSortedKeys();

			for (int i = 0; i < kwds.size(); i++) {
				String k1 = kwds.get(i);
				double cnt1 = c.getCount(k1);

				if (!targets.contains(k1)) {
					continue;
				}

				for (int j = 0; j < kwds.size(); j++) {
					String k2 = kwds.get(j);
					double cnt2 = c.getCount(k2);
					if (i == j) {
						continue;
					}
					cmm.incrementCount(k1, k2, type, Math.min(cnt1, cnt2));
				}
			}
		}

		{
			TextFileWriter writer = new TextFileWriter(KPPath.DATA_DIR + "scenario/kwds_target_cooccur.txt");

			for (String k1 : cmm.getOutKeyCountSums().getSortedKeys()) {
				CounterMap<String, String> cm = cmm.getCounterMap(k1);

				for (String k2 : cm.getOutKeyCountSums().getSortedKeys()) {
					Counter<String> c = cm.getCounter(k2);
					writer.write(String.format("%s\t%s\t%s\n", k1, k2, c.toString(c.size())));
				}
			}
			writer.close();
		}
	}

	public void getCocounts2() throws Exception {
		Set<String> targets = Generics
				.newHashSet(FileUtils.readLinesFromText(KPPath.DATA_DIR + "scenario/kwds_target.txt"));

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (String line : FileUtils.readLinesFromText(KPPath.DATA_DIR + "scenario/doc_kwds.txt")) {
			String[] ps = line.split("\t");
			String type = ps[0];
			String year = ps[1];
			Counter<String> c = Generics.newCounter(ps.length - 2);

			for (int i = 2; i < ps.length; i++) {
				String p = ps[i];
				String[] two = StrUtils.split2Two(":", p);
				c.incrementCount(two[0], Double.parseDouble(two[1]));
			}

			List<String> kwds = c.getSortedKeys();

			for (int i = 0; i < kwds.size(); i++) {
				String k1 = kwds.get(i);
				double cnt1 = c.getCount(k1);

				if (!targets.contains(k1)) {
					continue;
				}

				for (int j = 0; j < kwds.size(); j++) {
					String k2 = kwds.get(j);
					double cnt2 = c.getCount(k2);
					if (i == j) {
						continue;
					}
					String key = String.format("%s\t%s\t%s", k1, k2, year);
					cm.incrementCount(key, type, Math.min(cnt1, cnt2));
				}
			}
		}

		{
			CounterMap<String, String> cm2 = Generics.newCounterMap();

			for (String key : Generics.newTreeSet(cm.keySet())) {
				List<String> ps = StrUtils.split("\t", key);
				String k1 = ps.get(0);
				String k2 = ps.get(1);
				String year = ps.get(2);

				for (Entry<String, Double> e : cm.getCounter(key).entrySet()) {
					String type = e.getKey();
					double cnt = e.getValue();

					String key2 = String.format("%s\t%s\t%s", k1, k2, type);
					cm2.incrementCount(key2, year, cnt);
				}
			}

			cm = cm2;
		}

		{
			TextFileWriter writer = new TextFileWriter(KPPath.DATA_DIR + "scenario/kwds_target_cooccur_2.txt");

			for (String k1 : Generics.newTreeSet(cm.keySet())) {
				Counter<String> c = cm.getCounter(k1);
				writer.write(k1 + "\t" + c.toString(c.size()) + "\n");
			}
			writer.close();
		}
	}

	public void getYearCocounts2() throws Exception {

		Set<String> targets = Generics.newHashSet();

		for (String line : FileUtils.readLinesFromText(KPPath.DATA_DIR + "scenario/kwds_target_cooccur.txt")) {
			String[] ps = line.split("\t");
			targets.add(ps[0]);
			targets.add(ps[1]);
		}

		CounterMapMap<String, String, String> cmm = Generics.newCounterMapMap();

		for (String line : FileUtils.readLinesFromText(KPPath.DATA_DIR + "scenario/doc_kwds.txt")) {
			String[] ps = line.split("\t");
			String type = ps[0];
			String year = ps[1];
			Counter<String> c = Generics.newCounter(ps.length - 2);

			for (int i = 2; i < ps.length; i++) {
				String p = ps[i];
				String[] two = StrUtils.split2Two(":", p);
				c.incrementCount(two[0], Double.parseDouble(two[1]));
			}

			List<String> kwds = c.getSortedKeys();

			for (int i = 0; i < kwds.size(); i++) {
				String k1 = kwds.get(i);
				double cnt1 = c.getCount(k1);

				if (!targets.contains(k1)) {
					continue;
				}

				cmm.incrementCount(k1, type, year, cnt1);
			}
		}

		{
			TextFileWriter writer = new TextFileWriter(KPPath.DATA_DIR + "scenario/kwds_target_year.txt");

			for (String k1 : cmm.getOutKeyCountSums().getSortedKeys()) {
				CounterMap<String, String> cm = cmm.getCounterMap(k1);

				for (String k2 : cm.getOutKeyCountSums().getSortedKeys()) {
					Counter<String> c = cm.getCounter(k2);
					writer.write(String.format("%s\t%s\t%s\n", k1, k2, c.toString(c.size())));
				}
			}
			writer.close();
		}
	}

	public void getYearCocounts() throws Exception {

		Set<String> targets = Generics.newHashSet();

		for (String line : FileUtils.readLinesFromText(KPPath.DATA_DIR + "scenario/kwds_target_cooccur.txt")) {
			String[] ps = line.split("\t");
			targets.add(ps[0]);
			targets.add(ps[1]);
		}

		CounterMapMap<String, String, String> cmm = Generics.newCounterMapMap();

		for (String line : FileUtils.readLinesFromText(KPPath.DATA_DIR + "scenario/doc_kwds.txt")) {
			String[] ps = line.split("\t");
			String type = ps[0];
			String year = ps[1];
			Counter<String> c = Generics.newCounter(ps.length - 2);

			for (int i = 2; i < ps.length; i++) {
				String p = ps[i];
				String[] two = StrUtils.split2Two(":", p);
				c.incrementCount(two[0], Double.parseDouble(two[1]));
			}

			List<String> kwds = c.getSortedKeys();

			for (int i = 0; i < kwds.size(); i++) {
				String k1 = kwds.get(i);
				double cnt1 = c.getCount(k1);

				if (!targets.contains(k1)) {
					continue;
				}

				cmm.incrementCount(k1, type, year, cnt1);
			}
		}

		{
			TextFileWriter writer = new TextFileWriter(KPPath.DATA_DIR + "scenario/kwds_target_year.txt");

			for (String k1 : cmm.getOutKeyCountSums().getSortedKeys()) {
				CounterMap<String, String> cm = cmm.getCounterMap(k1);

				for (String k2 : cm.getOutKeyCountSums().getSortedKeys()) {
					Counter<String> c = cm.getCounter(k2);
					writer.write(String.format("%s\t%s\t%s\n", k1, k2, c.toString(c.size())));
				}
			}
			writer.close();
		}
	}

	public void format() throws Exception {
		CounterMap<String, String> kwdTypeWeights = FileUtils
				.readStringCounterMapFromText(KPPath.DATA_DIR + "scenario/kwds_tfidf.txt", false);
		CounterMap<String, String> kwdYearCnts = FileUtils
				.readStringCounterMapFromText(KPPath.DATA_DIR + "scenario/kwds_cnt_year.txt", false);

		// List<String> years = Generics.newArrayList(kwdYearCnts.innerKeySet());
		// Collections.sort(years);

		List<String> years = Generics.newArrayList();
		int base = 2007;

		for (int i = 0; i < 10; i++) {
			int y = base + i;
			years.add(y + "");
		}

		List<String> kwds = Generics.newArrayList();

		{
			Counter<String> c1 = kwdYearCnts.getOutKeyCountSums();

			c1.pruneKeysBelowThreshold(10);

			Counter<String> c2 = kwdTypeWeights.getOutKeyCountSums();
			c2.pruneExcept(c1.keySet());

			kwds = c2.getSortedKeys();
		}

		TextFileWriter writer = new TextFileWriter(KPPath.DATA_DIR + "scenario/kwds_cnt_year_2.txt");
		writer.write("KWD\t" + StrUtils.join("\t", years));

		for (int i = 0; i < kwds.size(); i++) {
			String kwd = kwds.get(i);
			Counter<String> c = kwdYearCnts.getCounter(kwd);

			StringBuffer sb = new StringBuffer();
			sb.append(kwd);

			for (int j = 0; j < years.size(); j++) {
				String year = years.get(j);
				int cnt = (int) c.getCount(year);
				sb.append(String.format("\t%d", cnt));
			}
			writer.write("\n" + sb.toString());
		}
		writer.close();
	}

	public void extractNtisText() throws Exception {

		FileUtils.deleteFilesUnder(KPPath.COL_LINE_DIR + "ntis/");

		List<String> targets = Generics.newArrayList();
		targets.add("국문과제명");
		targets.add("영문과제명");
		targets.add("연구목표요약");
		targets.add("기대효과요약");
		targets.add("연구내용요약");
		targets.add("한글키워드");
		targets.add("영문키워드");

		List<String> outs = Generics.newLinkedList();
		int batch_size = 20000;
		int batch_cnt = 0;

		for (File file : FileUtils.getFilesUnder(KPPath.COL_DB_DIR + "ntis/NTIS_과제정보")) {
			if (!file.getName().endsWith("txt")) {
				continue;
			}

			List<String> lines = FileUtils.readLinesFromText(file);

			List<String> attrs = StrUtils.split("\t", lines.get(0));
			attrs = StrUtils.unwrap(attrs);

			Map<String, Integer> m = Generics.newHashMap();

			for (int i = 0; i < attrs.size(); i++) {
				m.put(attrs.get(i), i);
			}

			for (int i = 1; i < lines.size(); i++) {
				List<String> vals = StrUtils.split("\t", lines.get(i));
				vals = StrUtils.unwrap(vals);

				if (attrs.size() != vals.size()) {
					continue;
				}

				String type = "ntis";
				String cn = "";
				String korTitle = "";
				String engTitle = "";
				String korAbs = "";
				String engAbs = "";
				String korKwdStr = "";
				String engKwdStr = "";
				String date = "";

				korTitle = vals.get(m.get("국문과제명"));
				engTitle = vals.get(m.get("영문과제명"));

				date = vals.get(m.get("기준년도"));

				StringBuffer sb = new StringBuffer();

				sb.append(vals.get(m.get("연구목표요약")));
				sb.append("\n\n" + vals.get(m.get("기대효과요약")));
				sb.append("\n\n" + vals.get(m.get("연구내용요약")));

				korAbs = sb.toString().trim().replace("\n", "<nl>");

				korKwdStr = vals.get(m.get("한글키워드"));
				engKwdStr = vals.get(m.get("영문키워드") != null ? m.get("영문키워드") : m.get("영어키워드"));

				{
					List<String> l = Generics.newArrayList();
					for (String s : korKwdStr.split(",")) {
						l.add(s.trim());
					}
					korKwdStr = StrUtils.join("<nl>", l);
				}

				{
					List<String> l = Generics.newArrayList();
					for (String s : engKwdStr.split(",")) {
						l.add(s.trim());
					}
					engKwdStr = StrUtils.join("<nl>", l);
				}

				String[] ps = new String[] { type, cn, korKwdStr, engKwdStr, korTitle, engTitle, korAbs, engAbs, date };
				ps = StrUtils.wrap(ps);

				String out = StrUtils.join("\t", ps);

				if (StrUtils.split("\t", out).size() != 9) {
					continue;
				}

				outs.add(out);

				if (outs.size() % batch_size == 0) {
					DecimalFormat df = new DecimalFormat("000000");

					String outFileName = KPPath.COL_LINE_DIR + "ntis/" + df.format(batch_cnt++) + ".txt.gz";

					FileUtils.writeStringCollectionAsText(outFileName, outs);
					outs.clear();
				}
			}
		}

		if (outs.size() > 0) {
			DecimalFormat df = new DecimalFormat("000000");
			String outFileName = KPPath.COL_LINE_DIR + "ntis/" + df.format(batch_cnt++) + "txt.gz";
			FileUtils.writeStringCollectionAsText(outFileName, outs);
			outs.clear();
		}
	}

	public void extractExcelFiles() throws Exception {
		for (File file : FileUtils.getFilesUnder(KPPath.COL_DB_DIR + "ntis/NTIS_과제정보")) {
			if (!file.getName().endsWith("xlsx")) {
				continue;
			}

			// if (!file.getName().contains("2005")) {
			// continue;
			// }

			FileInputStream fis = new FileInputStream(file);

			// Finds the workbook instance for XLSX file
			XSSFWorkbook myWorkBook = new XSSFWorkbook(fis); // Return first sheet from the XLSX workbook
			XSSFSheet mySheet = myWorkBook.getSheetAt(0);

			// Get iterator to all the rows in current sheet
			Iterator<Row> rowIterator = mySheet.iterator();

			List<String> outs = Generics.newLinkedList();

			// Traversing over each row of XLSX file
			while (rowIterator.hasNext()) {
				Row row = rowIterator.next();

				// For each row, iterate through each columns

				Iterator<Cell> cellIterator = row.cellIterator();

				List<String> vals = Generics.newLinkedList();

				while (cellIterator.hasNext()) {
					Cell cell = cellIterator.next();
					vals.add(cell.toString());

					switch (cell.getCellType()) {
					case Cell.CELL_TYPE_STRING:
						// System.out.print(cell.getStringCellValue() + "\t");
						break;
					case Cell.CELL_TYPE_NUMERIC:
						// System.out.print(cell.getNumericCellValue() + "\t");
						break;
					case Cell.CELL_TYPE_BOOLEAN:
						// System.out.print(cell.getBooleanCellValue() + "\t");
						break;
					default:
					}
				}

				vals = Generics.newArrayList(vals);

				for (int i = 0; i < vals.size(); i++) {
					vals.set(i, vals.get(i).replace("\r", "").replace("\n", "<nl>").replace("\t", " "));
				}

				vals = StrUtils.wrap(vals);
				outs.add(StrUtils.join("\t", vals));
			}

			FileUtils.writeStringCollectionAsText(file.getPath().replace(".xlsx", ".txt"), outs);
		}
	}

	private WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();

	private String indexPath = KPPath.COL_DIR + "lucene_idx";

	public void tokenize() throws Exception {
		List<File> files = FileUtils.getFilesUnder(KPPath.COL_LINE_DIR);

		for (File file : files) {
			List<String> ins = FileUtils.readLinesFromText(file);
			List<String> outs = Generics.newArrayList(ins.size());

			for (int u = 0; u < ins.size(); u++) {
				String line = ins.get(u);
				List<String> ps = StrUtils.split("\t", line);
				ps = StrUtils.unwrap(ps);

				String type = "";
				String cn = "";
				String korTitle = "";
				String engTitle = "";
				String korAbs = "";
				String engAbs = "";
				String korKwdStr = "";
				String engKwdStr = "";
				String date = "";

				{
					int i = 0;
					type = ps.get(i++);
					cn = ps.get(i++);
					korKwdStr = ps.get(i++);
					engKwdStr = ps.get(i++);
					korTitle = ps.get(i++);
					engTitle = ps.get(i++);
					korAbs = ps.get(i++);
					engAbs = ps.get(i++);
					date = ps.get(i++);
				}

				// if (date.length() == 0) {
				// continue;
				// }
				//
				// if (date.length() == 8) {
				//
				// } else if (date.length() == 6) {
				// date = date + "00";
				// } else if (date.length() == 4) {
				// date = date + "0000";
				// } else {
				// date = date.replace("-", "");
				// }

				if (korTitle.length() == 0 && korAbs.length() == 0) {
					continue;
				}

				korTitle = preprocess(korTitle.replace("<nl>", "\n"));
				korAbs = preprocess(korAbs.replace("<nl>", "\n"));

				korTitle = korTitle.replace("\n", "<nl>");
				korAbs = korAbs.replace("\n", "<nl>");

				List<String> l = StrUtils.split("<nl>", korKwdStr);

				for (int i = 0; i < l.size(); i++) {
					l.set(i, preprocess(l.get(i)));
				}

				korKwdStr = StrUtils.join("<nl>", l);

				ps.set(2, korKwdStr);
				ps.set(4, korTitle);
				ps.set(6, korAbs);

				ps = StrUtils.wrap(ps);

				String out = StrUtils.join("\t", ps);

				if (ps.size() != 9) {
					continue;
				}

				outs.add(out);
			}

			FileUtils.writeStringCollectionAsText(file.getPath().replace("line", "line_2"), outs);
		}

	}

	public void index() throws Exception {
		List<File> files = FileUtils.getFilesUnder(KPPath.COL_LINE_DIR);

		Directory dir = FSDirectory.open(Paths.get(indexPath));
		// Analyzer analyzer = new StandardAnalyzer();
		IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
		boolean create = true;

		if (create) {
			// Create a new index in the directory, removing any
			// previously indexed documents:
			iwc.setOpenMode(OpenMode.CREATE);
		} else {
			// Add new documents to an existing index:
			iwc.setOpenMode(OpenMode.CREATE_OR_APPEND);
		}

		// Optional: for better indexing performance, if you
		// are indexing many documents, increase the RAM
		// buffer. But if you do this, increase the max heap
		// size to the JVM (eg add -Xmx512m or -Xmx1g):
		//
		iwc.setRAMBufferSizeMB(1024.0);

		IndexWriter iw = new IndexWriter(dir, iwc);

		for (File file : files) {
			List<String> lines = FileUtils.readLinesFromText(file);

			for (String line : lines) {
				List<String> ps = StrUtils.split("\t", line);
				ps = StrUtils.unwrap(ps);

				String type = "";
				String cn = "";
				String korTitle = "";
				String engTitle = "";
				String korAbs = "";
				String engAbs = "";
				String korKwdStr = "";
				String engKwdStr = "";
				String date = "";

				{
					int i = 0;
					type = ps.get(i++);
					cn = ps.get(i++);
					korKwdStr = ps.get(i++);
					engKwdStr = ps.get(i++);
					korTitle = ps.get(i++);
					engTitle = ps.get(i++);
					korAbs = ps.get(i++);
					engAbs = ps.get(i++);
					date = ps.get(i++);
				}

				if (date.length() == 0) {
					continue;
				}

				if (date.length() == 8) {

				} else if (date.length() == 6) {
					date = date + "00";
				} else if (date.length() == 4) {
					date = date + "0000";
				} else {
					date = date.replace("-", "");
				}

				if (korTitle.length() == 0 && korAbs.length() == 0) {
					continue;
				}

				String content = korTitle + "<nl>" + korAbs;
				content = preprocess(content);

				Document d = new Document();
				// d.add(new TextField("contents", new StringReader(content)));
				d.add(new TextField("contents", content, Store.YES));
				d.add(new StringField("date", date, Store.YES));
				d.add(new StringField("cn", cn, Store.YES));
				d.add(new StringField("type", type, Store.YES));
				iw.addDocument(d);
			}
		}

		iw.close();
	}

	public String preprocess(String content) {
		StringBuffer sb = new StringBuffer();

		content = content.replace("<nl>", "\n");
		content = content.replace(". ", ".\n\n");

		for (String sent : content.split("\n")) {
			if (sent.length() == 0) {
				continue;
			}

			MSentence s = new MSentence();

			for (LNode node : Analyzer.parseJava(sent)) {
				Morpheme m = node.morpheme();
				WrappedArray<String> fs = m.feature();
				String[] vals = (String[]) fs.array();
				String word = m.surface();
				String pos = vals[0];

				MToken t = new MToken(2);
				t.add(word);
				t.add(pos);
				s.add(t);
			}
			sb.append(StrUtils.join(" ", s.getTokenStrings(0)) + "\n");
		}
		return sb.toString().trim();
	}

	public void search() throws Exception {
		IndexReader ir = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
		IndexSearcher is = new IndexSearcher(ir);

		BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));

		String SEARCH_FIELD = "contents";

		QueryParser parser = new QueryParser(SEARCH_FIELD, analyzer);

		while (true) {
			System.out.println("Enter query: ");
			String q = in.readLine().trim();

			if (q == null || q.equals("exit")) {
				break;
			}

			if (q.length() == 0) {
				continue;
			}

			q = preprocess(q);

			Query query = parser.parse(q);
			System.out.println("Searching for: " + query.toString(SEARCH_FIELD));

			TopDocs results = is.search(query, 10);
			ScoreDoc[] hits = results.scoreDocs;

			for (int i = 0; i < hits.length; i++) {
				int did = hits[i].doc;
				double score = hits[i].score;
				Document doc = is.doc(did);

				String date = doc.get("date");
				String type = doc.get("type");
				String cn = doc.get("cn");
				String contents = doc.get("contents");

				StringBuffer sb = new StringBuffer();
				sb.append(String.format("docid:\t%d", did));
				sb.append(String.format("\nscore:\t%f", score));
				sb.append(String.format("\ncn:\t%s", cn));
				sb.append(String.format("\ntype:\t%s", type));
				sb.append(String.format("\ndate:\t%s", date));
				sb.append(String.format("\ncontents:\t\n%s", contents));

				System.out.println(sb.toString() + "\n");

			}

		}
		ir.close();
	}

}

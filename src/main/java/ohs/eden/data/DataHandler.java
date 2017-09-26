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
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
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
		// dh.extractText();
		dh.tokenize();

		System.out.println("process ends.");
	}

	public void extractText() throws Exception {

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

		for (File file : FileUtils.getFilesUnder(KPPath.COL_DB_DIR + "ntis/ntis_과제정보")) {
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
				outs.add(StrUtils.join("\t", ps));

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
		for (File file : FileUtils.getFilesUnder(KPPath.COL_DB_DIR + "ntis/ntis_과제정보")) {
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
			List<String> lines = FileUtils.readLinesFromText(file);

			for (int u = 0; u < lines.size(); u++) {
				String line = lines.get(u);
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

//				if (date.length() == 0) {
//					continue;
//				}
//
//				if (date.length() == 8) {
//
//				} else if (date.length() == 6) {
//					date = date + "00";
//				} else if (date.length() == 4) {
//					date = date + "0000";
//				} else {
//					date = date.replace("-", "");
//				}

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

				lines.set(u, StrUtils.join("\t", ps));
			}

			FileUtils.writeStringCollectionAsText(file.getPath().replace("line", "line_2"), lines);
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

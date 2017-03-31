package ohs.ir.medical.general;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.FSDirectory;

import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.ir.lucene.common.AnalyzerUtils;
import ohs.ir.lucene.common.CommonFieldNames;
import ohs.ir.lucene.common.MedicalEnglishAnalyzer;
import ohs.ir.lucene.common.MyTextField;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * Construct an inverted index with source document collection.
 * 
 * @author Heung-Seon Oh
 * 
 */
public class DocumentIndexer {

	public static final int ram_size = 5000;

	public static IndexWriter getIndexWriter(String outputDirName) throws Exception {
		FileUtils.deleteFilesUnder(outputDirName);

		IndexWriterConfig iwc = new IndexWriterConfig(MedicalEnglishAnalyzer.newAnalyzer());
		// IndexWriterConfig iwc = new IndexWriterConfig(new
		// StandardAnalyzer());
		iwc.setOpenMode(OpenMode.CREATE);
		iwc.setRAMBufferSizeMB(ram_size);
		IndexWriter ret = new IndexWriter(FSDirectory.open(Paths.get(outputDirName)), iwc);
		return ret;
	}

	public static Set<String> getStopSectionNames() {
		String[] stopSectionNames = { "references", "external links", "see also", "notes", "further reading" };
		Set<String> ret = new HashSet<String>();
		for (String s : stopSectionNames) {
			ret.add(s);
		}
		return ret;
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		DocumentIndexer di = new DocumentIndexer();
		// di.indexTrecCds();
		di.indexOhsumed();
		// di.indexClefEHealth();
		// di.indexTrecGenomics();
		// di.indexWiki();
		// di.indexClueWeb12();
		// di.makeDocumentIdMap();

		System.out.println("process ends.");
	}

	public DocumentIndexer() {

	}

	public void indexClefEHealth() throws Exception {
		System.out.println("index CLEF eHealth.");
		IndexWriter iw = getIndexWriter(MIRPath.CLEF_EH_2014_INDEX_DIR);

		for (File file : FileUtils.getFilesUnder(MIRPath.CLEF_EH_2014_COL_LINE_DIR)) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] parts = line.split("\t");

				parts = StrUtils.unwrap(parts);

				String uid = parts[0];
				String date = parts[1];
				String url = parts[2];
				String content = parts[3].replaceAll("\\n", "\n");

				Document doc = new Document();
				doc.add(new StringField(CommonFieldNames.DOCUMENT_ID, uid, Field.Store.YES));
				doc.add(new StringField(CommonFieldNames.URL, url, Field.Store.YES));
				doc.add(new StringField(CommonFieldNames.DATE, date, Field.Store.YES));
				doc.add(new MyTextField(CommonFieldNames.CONTENT, content, Store.YES));

				iw.addDocument(doc);
			}
		}
		iw.close();
	}

	public void indexClueWeb12() throws Exception {
		System.out.println("index ClueWeb12.");

		IndexWriter iw = getIndexWriter(MIRPath.CLUEWEB_INDEX_DIR);

		List<File> files = FileUtils.getFilesUnder(MIRPath.CLUEWEB_COL_LINE_DIR);

		Collections.sort(files);

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			List<String> lines = FileUtils.readLinesFromText(file.getPath());

			for (int j = 1; j < lines.size(); j++) {
				String line = lines.get(j);
				String[] parts = line.split("\t");

				if (parts.length > 4) {
					String ss = StrUtils.join("", parts, 3);

					parts = new String[] { parts[0], parts[1], parts[2], ss };
				}

				if (parts.length != 4) {
					System.out.printf("[%s]\n", line);
					continue;
				}

				if (parts[0].length() == 0 || parts[1].length() == 0 || parts[2].length() == 0) {
					System.out.printf("[%s]\n", line);
					continue;
				}

				parts = StrUtils.unwrap(parts);

				String id = parts[0];
				String text = parts[1];
				String uri = parts[2];
				String linkStr = parts[3];

				StringBuffer sb = new StringBuffer();

				String[] ss = text.split("<nl>");
				for (int k = 0; k < ss.length; k++) {
					String[] toks = ss[k].split(" ");
					for (int l = 0; l < toks.length; l++) {
						String tok = toks[l].trim();
						if (tok.length() > 0) {
							if (tok.startsWith("tbi:")) {
								sb.append(tok.substring(4).replace("_", " "));
							} else {
								sb.append(tok);
							}
						}

						if (l != toks.length - 1) {
							sb.append(" ");
						}
					}

					if (k != ss.length - 1) {
						sb.append("\n");
					}
				}

				List<String> links = Generics.newArrayList();

				if (linkStr.length() > 0) {
					for (String link : linkStr.split("<tab>")) {
						int idx = link.indexOf(":");
						if (idx > -1) {
							String type = link.substring(0, idx);
							String url = link.substring(idx + 1);
							if (url.length() > 0 && url.startsWith("http")) {
								links.add(url);
							}
						}
					}
				}

				Document doc = new Document();
				doc.add(new StringField(CommonFieldNames.DOCUMENT_ID, id, Field.Store.YES));
				doc.add(new StringField(CommonFieldNames.URL, uri, Field.Store.YES));
				doc.add(new MyTextField(CommonFieldNames.CONTENT, sb.toString(), Store.YES));
				doc.add(new TextField(CommonFieldNames.LINKS, StrUtils.join("\n", links), Store.YES));

				iw.addDocument(doc);
			}
		}

		iw.close();

	}

	public void indexOhsumed() throws Exception {
		System.out.println("index OHSUMED.");

		IndexWriter iw = getIndexWriter(MIRPath.OHSUMED_INDEX_DIR);

		EnglishAnalyzer ea = new EnglishAnalyzer();

		for (File file : FileUtils.getFilesUnder(MIRPath.OHSUMED_COL_LINE_DIR)) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] parts = line.split("\t");
				parts = StrUtils.unwrap(parts);

				String seqId = parts[0];
				String medlineId = parts[1];
				String meshTerms = parts[2];
				String title = parts[3];
				String publicationType = parts[4];
				String abs = parts[5].replace(StrUtils.LINE_REP, "\n");
				String authors = parts[6];
				String source = parts[7];

				List<String> tt = AnalyzerUtils.getWords(abs, ea);
				
				System.out.println(abs);
				System.out.println(StrUtils.join(" ", tt));
				System.out.println();

				Document doc = new Document();
				doc.add(new StringField(CommonFieldNames.DOCUMENT_ID, medlineId, Field.Store.YES));
				doc.add(new MyTextField(CommonFieldNames.CONTENT, title + "\n" + abs, Field.Store.YES));
				iw.addDocument(doc);
			}
		}
		iw.close();
	}

	public void indexTrecCds() throws Exception {
		String[] inFileNames = { MIRPath.TREC_CDS_2014_COL_LINE_DIR, MIRPath.TREC_CDS_2016_COL_LINE_DIR };
		String[] outDirNames = { MIRPath.TREC_CDS_2014_INDEX_DIR, MIRPath.TREC_CDS_2016_INDEX_DIR };

		for (int i = 0; i < inFileNames.length; i++) {
			if (i != 0) {
				continue;
			}
			indexTrecCds(inFileNames[i], outDirNames[i]);
		}
	}

	public void indexTrecCds(String inDirName, String outDirName) throws Exception {
		System.out.println("index TREC CDS.");

		IndexWriter iw = getIndexWriter(outDirName);

		for (File file : FileUtils.getFilesUnder(inDirName)) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] parts = line.split("\t");

				if (parts.length != 5) {
					continue;
				}
				parts = StrUtils.unwrap(parts);

				String pmcId = parts[1];
				String title = parts[2];
				String abs = parts[3];
				String content = parts[4];
				content = title + "\n" + abs + "\n" + content;
				content = content.replace(StrUtils.LINE_REP, "\n");

				Document doc = new Document();
				doc.add(new StringField(CommonFieldNames.DOCUMENT_ID, pmcId, Field.Store.YES));
				doc.add(new TextField(CommonFieldNames.TITLE, title, Store.YES));
				doc.add(new MyTextField(CommonFieldNames.ABSTRACT, abs, Store.YES));
				doc.add(new MyTextField(CommonFieldNames.CONTENT, content, Store.YES));
				iw.addDocument(doc);
			}
		}
		iw.close();
	}

	public void indexTrecGenomics() throws Exception {
		IndexWriter iw = getIndexWriter(MIRPath.TREC_GENO_2007_INDEX_DIR);

		for (File file : FileUtils.getFilesUnder(MIRPath.TREC_GENO_2007_COL_LINE_DIR)) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] parts = line.split("\t");

				if (parts.length != 2) {
					continue;
				}

				parts = StrUtils.unwrap(parts);

				String id = parts[0];
				String content = parts[1];

				int start = id.lastIndexOf("/");
				int end = id.lastIndexOf(".");
				id = id.substring(start + 1, end);

				Document doc = new Document();
				doc.add(new StringField(CommonFieldNames.DOCUMENT_ID, id, Field.Store.YES));
				doc.add(new MyTextField(CommonFieldNames.CONTENT, content, Store.YES));

				iw.addDocument(doc);
			}
		}
		iw.close();
	}

	public void indexWiki() throws Exception {
		IndexWriter iw = getIndexWriter(MIRPath.WIKI_INDEX_DIR);
		for (File file : FileUtils.getFilesUnder(MIRPath.WIKI_COL_LINE_DIR)) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] parts = line.split("\t");

				if (parts.length != 2) {
					continue;
				}

				parts = StrUtils.unwrap(parts);

				String title = parts[0];
				String content = parts[1].replace(StrUtils.LINE_REP, "\n");

				Document doc = new Document();
				doc.add(new StringField(CommonFieldNames.TITLE, title, Store.YES));
				doc.add(new MyTextField(CommonFieldNames.CONTENT, content, Store.YES));
				iw.addDocument(doc);
			}
		}
		iw.close();
	}

	public void makeDocumentIdMap() throws Exception {
		String[] indexDirNames = MIRPath.IndexDirNames;
		String[] docMapFileNames = MIRPath.DocIdMapFileNames;

		for (int i = 3; i < indexDirNames.length; i++) {
			String indexDirName = indexDirNames[i];
			String docMapFileName = docMapFileNames[i];

			System.out.printf("process [%s].\n", indexDirNames[i]);

			File outputFile = new File(docMapFileName);

			// if (outputFile.exists()) {
			// return;
			// }

			IndexSearcher is = SearcherUtils.getIndexSearcher(indexDirName);
			IndexReader ir = is.getIndexReader();

			List<String> docIds = new ArrayList<String>();

			for (int j = 0; j < ir.maxDoc(); j++) {
				if ((j + 1) % 100000 == 0) {
					System.out.printf("\r[%d/%d]", j + 1, ir.maxDoc());
				}
				Document doc = ir.document(j);
				String docId = doc.getField(CommonFieldNames.DOCUMENT_ID).stringValue();
				docIds.add(docId);
			}
			System.out.printf("\r[%d/%d]\n", ir.maxDoc(), ir.maxDoc());

			TextFileWriter writer = new TextFileWriter(docMapFileName);
			for (int j = 0; j < docIds.size(); j++) {
				String output = j + "\t" + docIds.get(j);
				writer.write(output + "\n");
			}
			writer.close();
		}
	}

}

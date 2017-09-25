package ohs.eden.data;

import java.io.File;
import java.io.StringReader;
import java.nio.file.Paths;
import java.util.List;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.bitbucket.eunjeon.seunjeon.Analyzer;
import org.bitbucket.eunjeon.seunjeon.LNode;
import org.bitbucket.eunjeon.seunjeon.Morpheme;

import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.utils.StrUtils;
import scala.collection.mutable.WrappedArray;

public class DataHandler {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// filter();

		index();

		System.out.println("process ends.");
	}

	public static void index() throws Exception {
		List<File> files = FileUtils.getFilesUnder(KPPath.COL_LINE_DIR);

		String indexPath = KPPath.COL_DIR + "lucene_idx";

		Directory dir = FSDirectory.open(Paths.get(indexPath));
		// Analyzer analyzer = new StandardAnalyzer();
		WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
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
				} else {
					date = date.replace("-", "");
				}

				if (korTitle.length() == 0 && korAbs.length() == 0) {
					continue;
				}

				String content = korTitle + "<nl>" + korAbs;
				content = content.replace("<nl>", "\n");
				content = content.replace(". ", ".\n\n");

				StringBuffer sb = new StringBuffer();

				int word_cnt = 0;

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
					word_cnt += s.size();
					sb.append(StrUtils.join(" ", s.getTokenStrings(0)) + "\n");
				}

				Document d = new Document();
				d.add(new TextField("contents", new StringReader(sb.toString().trim())));
				d.add(new StringField("date", date, Store.YES));
				d.add(new StringField("cn", cn, Store.YES));
				d.add(new StringField("type", type, Store.YES));
				iw.addDocument(d);
			}
		}

		iw.close();

	}

}

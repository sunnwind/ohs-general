package ohs.corpus.dump;

import java.io.File;
import java.io.StringReader;
import java.text.DecimalFormat;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.sweble.wikitext.engine.PageId;
import org.sweble.wikitext.engine.PageTitle;
import org.sweble.wikitext.engine.WtEngineImpl;
import org.sweble.wikitext.engine.config.WikiConfig;
import org.sweble.wikitext.engine.nodes.EngPage;
import org.sweble.wikitext.engine.nodes.EngProcessedPage;
import org.sweble.wikitext.engine.utils.DefaultConfigEnWp;
import org.sweble.wikitext.example.TextConverter;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * @author ohs
 * 
 */
public class WikiDumperOld extends TextDumper {

	public class Worker implements Callable<Integer> {

		// Set-up a simple wiki configuration
		private WikiConfig config = DefaultConfigEnWp.generate();
		// Instantiate a compiler for wiki pages
		private WtEngineImpl engine = new WtEngineImpl(config);
		// Retrieve a page

		private TextConverter p = new TextConverter(config, 10000);

		private File[] files;

		private AtomicInteger file_cnt;

		public Worker(File[] files, AtomicInteger file_cnt) {
			super();
			this.files = files;
			this.file_cnt = file_cnt;
		}

		@Override
		public Integer call() throws Exception {
			int file_loc = 0;

			while ((file_loc = file_cnt.getAndIncrement()) < files.length) {
				File file = files[file_loc];
				List<String> lines = Generics.newArrayList();

				for (String line : FileUtils.readLinesFromText(file)) {
					String[] parts = line.split("\t");

					parts = StrUtils.unwrap(parts);

					if (parts[1].toLowerCase().startsWith("#")) {
						continue;
					}

					String title = parts[0];
					String wikiText = parts[1];

					try {
						PageTitle pageTitle = PageTitle.make(config, title);
						PageId pageId = new PageId(pageTitle, -1);
						EngProcessedPage cp = engine.postprocess(pageId, wikiText, null);
						EngPage ep = cp.getPage();
						String s = (String) p.go(ep);
						parts[0] = title.replaceAll("[\n]+", StrUtils.LINE_REP);
						parts[1] = s.replaceAll("[\n]+", StrUtils.LINE_REP);

						parts = StrUtils.wrap(parts);

						lines.add(StrUtils.join("\t", parts));
					} catch (Exception e) {

					}
				}

				String outFileName = file.getPath().replace("tmp_line", "line");

				FileUtils.writeStringCollectionAsText(outFileName, lines);
			}

			return 0;
		}

	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		WikiDumperOld dh = new WikiDumperOld(MIRPath.WIKI_COL_XML_FILE, MIRPath.WIKI_COL_LINE_DIR);
		dh.dump();
		System.out.println("process ends.");
	}

	public static String[] parseXml(String text) throws Exception {
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder p1 = dbf.newDocumentBuilder();

		Document xmlDoc = p1.parse(new InputSource(new StringReader(text)));

		Element docElem = xmlDoc.getDocumentElement();

		String[] nodeNames = { "title", "text" };

		String[] values = new String[nodeNames.length];

		for (int j = 0; j < nodeNames.length; j++) {
			NodeList nodes = docElem.getElementsByTagName(nodeNames[j]);
			if (nodes.getLength() > 0) {
				values[j] = nodes.item(0).getTextContent().trim();
			}
		}
		return values;
	}

	private int thread_size = 1;

	public WikiDumperOld(String inputDir, String outputFileName) {
		super(inputDir, outputFileName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		extractLines();
		// extractText();

	}

	public void extractLines() throws Exception {
		String outDirName = this.outPathName.replace("line", "tmp_line");

		FileUtils.deleteFilesUnder(outDirName);

		DecimalFormat df = new DecimalFormat("00000");
		StringBuffer sb = new StringBuffer();
		boolean is_page = false;

		List<String> docs = Generics.newArrayList();
		int file_cnt = 0;
		int batch_size = 10000;

		TextFileReader reader = new TextFileReader(inPathName);
		reader.setPrintSize(100000);

		while (reader.hasNext()) {
			reader.printProgress();
			String line = reader.next();
			System.out.println(line);

			if (line.trim().startsWith("<page>")) {
				is_page = true;
				sb.append(line + "\n");
			} else if (line.trim().startsWith("</page>")) {
				sb.append(line);

				String xml = StrUtils.normalizeSpaces(sb.toString());

				// System.out.println(sb.toString() + "\n\n");
				String[] values = null;

				try {
					values = parseXml(xml);
				} catch (Exception e) {
					e.printStackTrace();
					continue;
				}

				values = StrUtils.wrap(values);
				docs.add(StrUtils.join("\t", values));

				sb = new StringBuffer();
				is_page = false;

				if (docs.size() % batch_size == 0) {
					String outFileName = outDirName + String.format("%s.txt.gz", df.format(file_cnt++));
					FileUtils.writeStringCollectionAsText(outFileName, docs);
					docs.clear();
				}
			} else {
				if (is_page) {
					sb.append(line + "\n");
				}
			}
		}
		reader.printProgress();
		reader.close();

		if (docs.size() > 0) {
			String outFileName = outDirName + String.format("%s.txt.gz", df.format(file_cnt++));
			FileUtils.writeStringCollectionAsText(outFileName, docs);
			docs.clear();
		}
	}

	public void extractText() throws Exception {
		File[] files = new File(outPathName.replace("line", "tmp_line")).listFiles();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		AtomicInteger file_cnt = new AtomicInteger(0);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(files, file_cnt)));
		}

		for (int i = 0; i < fs.size(); i++) {
			fs.get(i).get();
		}

		tpe.shutdown();

	}

}

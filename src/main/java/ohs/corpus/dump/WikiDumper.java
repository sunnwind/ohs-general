package ohs.corpus.dump;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.select.NodeTraversor;
import org.jsoup.select.NodeVisitor;

import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

/**
 * @author ohs
 * 
 */
public class WikiDumper extends TextDumper {

	static boolean lastCharIsWhitespace(StringBuilder sb) {
		return sb.length() != 0 && sb.charAt(sb.length() - 1) == ' ';
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		WikiDumper dh = new WikiDumper(MIRPath.WIKI_COL_DIR + "raw/", MIRPath.WIKI_COL_LINE_DIR);
		dh.dump();
		System.out.println("process ends.");
	}

	public static String text(Element elem) {
		final StringBuilder accum = new StringBuilder();
		new NodeTraversor(new NodeVisitor() {
			public void head(Node node, int depth) {
				if (node instanceof TextNode) {
					TextNode textNode = (TextNode) node;
					accum.append(textNode.getWholeText());
				} else if (node instanceof Element) {
					Element element = (Element) node;

					if (accum.length() > 0 && (element.isBlock() || elem.tag().getName().equals("br")) && !lastCharIsWhitespace(accum))
						accum.append(" ");
				}
			}

			public void tail(Node node, int depth) {
			}
		}).traverse(elem);
		return accum.toString().trim();
	}

	private int thread_size = 1;

	public WikiDumper(String inDirName, String outFileName) {
		super(inDirName, outFileName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		extractLines();

	}

	public void extractLines() throws Exception {
		FileUtils.deleteFilesUnder(outPathName);

		int page_cnt = 0;
		int file_cnt = 0;
		int batch_size = 5000;

		Timer timer = Timer.newTimer();

		List<File> files = FileUtils.getFilesUnder(inPathName);

		TextFileWriter writer = new TextFileWriter(MIRPath.WIKI_DIR + "phrss.txt.gz");

		DecimalFormat df = new DecimalFormat("0000000");

		List<String> lines = Generics.newLinkedList();

		for (int i = 0; i < files.size(); i++) {
			File file = files.get(i);
			String content = FileUtils.readFromText(file);

			Document doc = Jsoup.parse(content);

			org.jsoup.select.Elements docElems = doc.getElementsByTag("doc");

			for (int j = 0; j < docElems.size(); j++) {
				Element docElem = docElems.get(j);

				String id = docElem.attr("id");
				String url = docElem.attr("url");
				String title = docElem.attr("title");

				List<String> phrss = Generics.newArrayList();
				phrss.add(String.format("Title:\t%s", title));

				org.jsoup.select.Elements linkElems = docElem.getElementsByTag("a");

				for (int k = 0; k < linkElems.size(); k++) {
					Element linkElem = linkElems.get(k);
					String href = linkElem.attr("href");
					phrss.add(linkElem.text());
				}

				String phrsOut = StrUtils.join("\n", phrss);

				if (++page_cnt > 1) {
					phrsOut = "\n\n" + phrsOut;
				}

				writer.write(phrsOut);

				String docText = text(docElem);

				docText = docText.replaceAll("\n", "<nl>");

				List<String> items = Generics.newArrayList();
				items.add(id);
				items.add(url);
				items.add(title);
				items.add(docText);
				items.add(StrUtils.join("|", phrss.subList(1, phrss.size())));

				items = StrUtils.wrap(items);

				lines.add(StrUtils.join("\t", items));

				if (lines.size() % batch_size == 0) {
					File outFile = new File(outPathName, df.format(++file_cnt) + ".txt.gz");
					FileUtils.writeStringCollection(outFile.getPath(), lines);
					lines = Generics.newLinkedList();
				}

				if (page_cnt % 10000 == 0) {
					System.out.printf("[%d, %s]\n", page_cnt, timer.stop());
				}
			}
		}
		writer.close();

		System.out.printf("[%d, %s]\n", page_cnt, timer.stop());

		if (lines.size() > 0) {
			File outFile = new File(outPathName, df.format(++file_cnt) + ".txt.gz");
			FileUtils.writeStringCollection(outFile.getPath(), lines);
			lines = Generics.newArrayList();
		}
	}

}

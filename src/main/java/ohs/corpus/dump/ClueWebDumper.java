package ohs.corpus.dump;

import java.io.File;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.safety.Whitelist;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.io.TextFileWriter;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class ClueWebDumper extends TextDumper {

	public class DumpWorker implements Callable<Integer> {

		public DumpWorker() {
		}

		@Override
		public Integer call() {
			int cur_cnt = 0;
			Timer timer = Timer.newTimer();

			while ((cur_cnt = file_cnt.getAndIncrement()) < files.size()) {
				File file = files.get(cur_cnt);

				if (!file.getName().endsWith(".gz")) {
					continue;
				}

				// fileNameWriter.write(file.getPath() + "\n");

				String outFileName1 = String.format("%s/%s",

						file.getParent().replace("raw", "line"),

						file.getName().replace("warc", "txt"));

				if (FileUtils.exists(outFileName1)) {
					continue;
				}

				String outFileName2 = outFileName1.replace("line", "line_id").replace(".gz", "");

				Set<String> stopIds = Generics.newHashSet();

				if (FileUtils.exists(outFileName2)) {
					Set<String> starts = Generics.newHashSet();
					Set<String> ends = Generics.newHashSet();

					TextFileReader reader = new TextFileReader(outFileName2);

					while (reader.hasNext()) {
						String[] parts = reader.next().split("\t");
						if (parts[0].startsWith("START")) {
							starts.add(parts[1]);
						} else {
							ends.add(parts[1]);
						}
					}
					reader.close();

					for (String id : starts) {
						if (!ends.contains(id)) {
							stopIds.add(id);
						}
					}
				}

				Whitelist whitelist = Whitelist.relaxed();

				TextFileWriter writer = new TextFileWriter(outFileName2);
				TextFileReader reader = new TextFileReader(file);

				List<String> results = Generics.newLinkedList();
				List<String> lines = Generics.newLinkedList();
				int cnt = 0;

				while (reader.hasNext()) {
					String line = reader.next();

					if (line.startsWith("WARC/1.0")) {
						if (lines.size() > 0 && cnt > 1) {
							int start = -1;
							String id = "";
							String uri = "";

							for (int i = 0; i < lines.size(); i++) {
								if (lines.get(i).startsWith("WARC-TREC-ID")) {
									id = lines.get(i).substring(13).trim();
								} else if (lines.get(i).startsWith("WARC-Target-URI")) {
									uri = lines.get(i).substring(16).trim();
								} else if (lines.get(i).startsWith("Content-Length")) {
									start = i + 2;
									break;
								}
							}

							for (int i = start; i < lines.size(); i++) {
								if (lines.get(i).trim().length() == 0) {
									start = i + 1;
									break;
								}
							}

							writer.write(String.format("START\t%s\n", id));

							if (!stopIds.contains(id)) {
								String text1 = StrUtils.join("\n", lines, start, lines.size()).trim();
								String text2 = Jsoup.clean(text1, whitelist);

								if (Jsoup.isValid(text2, whitelist)) {
									try {
										List<String> items = Generics.newLinkedList();
										List<String> links = Generics.newLinkedList();

										Document doc = Jsoup.parse(text2);

										goDown(doc, items, links, false, 1);

										if (id.length() > 0 && items.size() > 0) {
											String content = StrUtils.join(StrUtils.LINE_REP, items);
											// String s2 =
											// StrUtils.join(StrUtils.TAB_REP,
											// links);

											content = StrUtils.normalizeSpaces(content);

											if (content.length() > 0) {
												String[] parts = new String[] { id, content, uri };
												results.add(StrUtils.join("\t", StrUtils.wrap(parts)));
											}

										}
									} catch (Exception e) {
										lines = Generics.newArrayList();
										continue;
									}
									writer.write(String.format("END\t%s\n", id));
								}
							}
						}

						lines = Generics.newArrayList();
						cnt++;
					} else {
						lines.add(line);
					}
				}
				reader.close();
				writer.close();

				try {
					FileUtils.writeStringCollectionAsText(outFileName1, results);
				} catch (Exception e) {
					e.printStackTrace();
				}

				// System.out.printf("[%s, %d, %s]\n", file.getPath(), cnt,
				// timer.stop());

				if (cur_cnt % 100 == 0) {
					System.out.printf("[%d/%d, %s]\n", cur_cnt, files.size(), timer.stop());
				}
			}

			if (cur_cnt == files.size()) {
				System.out.printf("[%d/%d, %s]\n", cur_cnt, files.size(), timer.stop());
			}

			return 0;
		}

		private void goDown(Node node, List<String> strs, List<String> links, boolean is_table_item, int depth) {
			if (depth >= 1000) {
				return;
			}

			List<Node> childNodes = node.childNodes();

			if (node instanceof Element) {
				Element elem = (Element) node;

				String tagName = elem.tagName();
				String type = "e";
				String url = "";

				if (tagName.equals("a")) {
					type = "a";
					url = elem.attr("abs:href");
				} else if (tagName.equals("link")) {
					type = "link";
					url = elem.attr("abs:href");
				} else {
					if (elem.hasAttr("type")) {
						type = elem.attr("type");
					}

					if (elem.hasAttr("abs:src")) {
						url = elem.attr("abs:src");
					}
				}

				if (!type.equals("e")) {
					type = StrUtils.normalizeSpaces(type);
				}

				if (url.length() > 0) {
					url = StrUtils.normalizeSpaces(url);
					links.add(type + ":" + url);
				}

				if (tagName.equals("table") || tagName.equals("ul")) {
					is_table_item = true;
				}
			}

			if (node instanceof TextNode) {
				TextNode tn = (TextNode) node;
				String t = tn.text();

				t = t.replaceAll("[\n]+", StrUtils.LINE_REP);
				t = StrUtils.normalizeSpaces(t);

				if (t.length() > 0) {
					// if (is_table_item) {
					// t = t.replaceAll("[ ]+", "_");
					// t = "tbi:" + t;
					// }
					strs.add(t);
				}
			}

			for (int i = 0; i < childNodes.size(); i++) {
				Node child = childNodes.get(i);

				goDown(child, strs, links, is_table_item, depth + 1);

			}
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		ClueWebDumper dh = new ClueWebDumper(MIRPath.CLUEWEB_COL_RAW_DIR, MIRPath.CLUEWEB_COL_LINE_DIR);
		dh.setThreadSize(200);
		dh.dump();

		System.out.println("process ends.");
	}

	private int thread_size = 1;

	private AtomicInteger file_cnt;

	private List<File> files;

	public ClueWebDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);
	}

	@Override
	public void dump() throws Exception {
		files = Generics.newArrayList();

		for (File f1 : new File(inPathName).listFiles()) {
			if (f1.isDirectory() && f1.getPath().contains("ClueWeb12_")) {
				for (File f2 : f1.listFiles()) {
					for (File f3 : f2.listFiles()) {
						files.add(f3);
					}
				}
			}
		}

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		file_cnt = new AtomicInteger(0);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new DumpWorker()));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}

		tpe.shutdown();

	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}

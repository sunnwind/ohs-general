package ohs.corpus.dump;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class ClefEHealthDumper extends TextDumper {

	class Worker implements Callable<Integer> {

		private String inFileName;

		public Worker(String inFileName) {
			super();
			this.inFileName = inFileName;
		}

		@Override
		public Integer call() throws Exception {

			String outPath = new File(inFileName.replace(inPathName, outPathName)).getParent().replace("\\", "/");

			List<String> res = Generics.newLinkedList();

			ZipInputStream zis = new ZipInputStream(new FileInputStream(new File(inFileName)));
			BufferedReader br = new BufferedReader(new InputStreamReader(zis));
			ZipEntry ze = null;

			while ((ze = zis.getNextEntry()) != null) {
				if (ze.isDirectory()) {
					continue;
				}

				List<String> lines = Generics.newLinkedList();
				String line = null;

				while ((line = br.readLine()) != null) {
					if (line.equals("")) {
						continue;
					}

					lines.add(line);

					if (line.startsWith("#EOR")) {
						String uid = lines.get(0);
						String date = lines.get(1);
						String url = lines.get(2);
						String html = StrUtils.join("\n", lines, 4, lines.size() - 1);

						if (!uid.startsWith("#UID") || !date.startsWith("#DATE") || !url.startsWith("#URL")
								|| !lines.get(3).startsWith("#CONTENT")) {

							lines.clear();
							continue;
						}

						uid = uid.substring(6);
						date = date.substring(6);
						url = url.substring(5);

						if (uid.startsWith(".")) {
							System.out.println(uid);
						}

						if (docIdSet != null && docIdSet.contains(uid)) {
							lines.clear();
							continue;
						}

						Pattern p = Pattern.compile("\\.([a-z]+)$");
						Matcher m = p.matcher(url);

						if (m.find()) {
							String exp = m.group(1).toLowerCase();

							if (stopExpSet.contains(exp)) {
								lines.clear();
								continue;
							}
						}

						Document doc = Jsoup.parse(html);
						String content = doc.text();

						if (content == null) {
							content = "";
						} else {
							content = content.replace("\n", StrUtils.LINE_REP);
						}

						String[] values = new String[] { uid, date, url, content };

						for (int i = 0; i < values.length; i++) {
							values[i] = StrUtils.normalizeSpaces(values[i]);
						}

						values = StrUtils.wrap(values);

						res.add(StrUtils.join("\t", values));

						if (res.size() % batch_size == 0) {
							DecimalFormat df = new DecimalFormat("00000");
							String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(batch_cnt.getAndIncrement()));
							FileUtils.writeStringCollectionAsText(outFileName, res);

							res.clear();
						}

						lines.clear();
					}
				}
			}
			br.close();

			if (res.size() > 0) {
				DecimalFormat df = new DecimalFormat("00000");
				String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(batch_cnt.getAndIncrement()));
				FileUtils.writeStringCollectionAsText(outFileName, res);
			}

			return (int) 0;
		}
	}

	public static Set<String> getStopFileExtensions() {
		Set<String> ret = new HashSet<String>();
		ret.add("doc");
		ret.add("docx");
		ret.add("pdf");
		ret.add("swf");
		ret.add("ppt");
		ret.add("pptx");
		ret.add("png");
		ret.add("flv");
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		ClefEHealthDumper dh = new ClefEHealthDumper(MIRPath.CLEF_EH_2014_COL_RAW_DIR, MIRPath.CLEF_EH_2014_COL_LINE_DIR);
		dh.dump();

		System.out.println("process ends.");
	}

	private Set<String> docIdSet;

	private AtomicInteger batch_cnt;

	private Set<String> stopExpSet;

	public ClefEHealthDumper(String inputDir, String outDirName) {
		super(inputDir, outDirName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s]\n", inPathName, outPathName);

		FileUtils.deleteFilesUnder(outPathName);

		stopExpSet = getStopFileExtensions();

		File[] files = new File(inPathName).listFiles();

		int thread_size = files.length;

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		batch_cnt = new AtomicInteger(0);

		for (int i = 0; i < files.length; i++) {
			String inFileName = files[i].getPath().replace("\\", "/");
			fs.add(tpe.submit(new Worker(inFileName)));
		}

		for (int i = 0; i < fs.size(); i++) {
			fs.get(i).get();
		}

		tpe.shutdown();
	}

}

package ohs.corpus.dump;

import java.io.File;
import java.io.FileInputStream;
import java.text.DecimalFormat;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class TrecGenomicsDumper extends TextDumper {

	class Worker implements Callable<Integer> {

		public Worker() {
			super();
		}

		@Override
		public Integer call() throws Exception {

			int loc = 0;
			List<String> res = Generics.newArrayList();

			String outPath = new File(inFiles[0].getPath().replace("\\", "/").replace(inPathName, outPathName)).getParent().replace("\\",
					"/");

			while ((loc = file_cnt.getAndIncrement()) < inFiles.length) {
				String inFileName = inFiles[loc].getPath().replace("\\", "/");

				ZipInputStream zis = new ZipInputStream(new FileInputStream(new File(inFileName)));
				// BufferedReader br = new BufferedReader(new InputStreamReader(zis));
				ZipEntry ze = null;
				// read every single entry in TAR file
				while ((ze = zis.getNextEntry()) != null) {
					// the following two lines remove the .tar.gz extension for the folder name
					// System.out.println(entry.getName());

					if (ze.isDirectory()) {
						continue;
					}

					String fileName = ze.getName();
					String docid = FileUtils.removeExtension(new File(fileName).getName());
					StringBuffer sb = new StringBuffer();

					int c;

					while ((c = zis.read()) != -1) {
						sb.append((char) c);
					}

					if (sb.length() > 0) {
						Document doc = Jsoup.parse(sb.toString());
						String content = doc.text();

						if (content == null) {
							content = "";
						} else {
							content = content.replace("\r\n", StrUtils.LINE_REP);
						}

						String[] values = StrUtils.wrap(new String[] { docid, content });

						for (int i = 0; i < values.length; i++) {
							if (values[i] == null) {
								values[i] = "";
							}
							values[i] = StrUtils.normalizeSpaces(values[i]);
						}

						res.add(StrUtils.join("\t", values));

						if (res.size() % batch_size == 0) {
							// String outPath = new File(inFileName.replace(inPathName, outPathName)).getParent().replace("\\", "/");
							DecimalFormat df = new DecimalFormat("00000");
							String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(batch_cnt.getAndIncrement()));
							FileUtils.writeStringCollection(outFileName, res);

							res.clear();
						}
					}
				}
				zis.close();
			}

			if (res.size() > 0) {
				DecimalFormat df = new DecimalFormat("00000");
				String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(batch_cnt.getAndIncrement()));
				FileUtils.writeStringCollection(outFileName, res);
				res.clear();
			}

			return 0;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		TrecGenomicsDumper d = new TrecGenomicsDumper(MIRPath.TREC_GENO_2007_COL_RAW_DIR, MIRPath.TREC_GENO_2007_COL_LINE_DIR);
		d.dump();
		System.out.println("process ends.");
	}

	private File[] inFiles;

	private AtomicInteger file_cnt;

	public TrecGenomicsDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		FileUtils.deleteFilesUnder(outPathName);

		inFiles = new File(inPathName).listFiles();

		int thread_size = 3;

		batch_size = 2000;

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		batch_cnt = new AtomicInteger(0);
		file_cnt = new AtomicInteger(0);

		for (int i = 0; i < inFiles.length; i++) {
			fs.add(tpe.submit(new Worker()));
		}

		for (int i = 0; i < fs.size(); i++) {
			fs.get(i).get();
		}

		tpe.shutdown();

	}
}

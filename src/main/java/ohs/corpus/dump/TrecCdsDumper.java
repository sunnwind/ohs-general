package ohs.corpus.dump;

import java.io.File;
import java.io.FileInputStream;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.jsoup.Jsoup;
import org.jsoup.helper.StringUtil;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.parser.Parser;
import org.jsoup.select.Elements;
import org.jsoup.select.NodeTraversor;
import org.jsoup.select.NodeVisitor;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class TrecCdsDumper extends TextDumper {

	class Worker1 implements Callable<Integer> {

		private String inFileName;

		public Worker1(String inFileName) {
			super();
			this.inFileName = inFileName;
		}

		@Override
		public Integer call() throws Exception {
			String outPath = new File(inFileName.replace(inPathName, outPathName)).getParent().replace("\\", "/");
			List<String> res = Generics.newArrayList();

			TarArchiveInputStream tis = new TarArchiveInputStream(new GZIPInputStream(new FileInputStream(new File(inFileName))));
			TarArchiveEntry tae = null;

			int num_docs = 0;
			// read every single entry in TAR file
			while ((tae = tis.getNextTarEntry()) != null) {
				// the following two lines remove the .tar.gz extension for the folder name
				// System.out.println(entry.getName());

				// if (num_docs_in_coll == 40000) {
				// break;
				// }

				if (tae.isDirectory()) {
					continue;
				}

				String fileName = tae.getName();
				StringBuffer sb = new StringBuffer();

				int c;
				while ((c = tis.read()) != -1) {
					sb.append((char) c);
				}

				if (sb.length() > 0) {
					Document doc = Jsoup.parse(sb.toString(), "", Parser.xmlParser());

					String pmcid = "";
					String title = "";
					String abs = "";
					String body = "";
					String kwds = "";
					String jTitle = "";
					String year = "";
					String month = "";
					String pmid = "";
					String doi = "";
					String refs = "";

					{
						Elements elem = doc.getElementsByAttributeValue("pub-id-type", "pmc");
						if (elem.size() > 0) {
							pmcid = elem.get(0).text();
						}
					}

					{
						Elements elem = doc.getElementsByTag("journal-meta");

						if (elem.size() > 0) {
							jTitle = elem.get(0).getElementsByTag("journal-title").text();
						}
					}

					{
						Elements elem = doc.getElementsByTag("article-meta");

						if (elem.size() > 0) {

							{
								Elements elem2 = elem.get(0).getElementsByTag("article-id");
								if (elem2.size() > 0) {
									Elements elem3 = elem2.get(0).getElementsByAttribute("pub-id-type");
									for (int i = 0; i < elem3.size(); i++) {
										Element elem4 = elem3.get(i);
										String val = elem4.attr("pub-id-type");
										String text = elem4.text();
										if (val.equals("pmid")) {
											pmid = text;
										} else if (val.equals("pmc")) {
											pmcid = text;
										} else if (val.equals("doi")) {
											doi = text;
										}
									}
								}
							}

							{
								Elements elem2 = elem.get(0).getElementsByTag("article-title");

								if (elem2.size() > 0) {
									title = elem2.get(0).text();
								}
							}

							{
								Elements elem2 = doc.getElementsByTag("abstract");

								if (elem2.size() > 0) {
									abs = elem2.get(0).text();
								}
							}

							{
								Elements elem2 = doc.getElementsByTag("kwd-group");
								if (elem2.size() > 0) {
									Elements elem3 = elem2.get(0).getElementsByTag("kwd");
									List<String> l = Generics.newArrayList();
									for (int i = 0; i < elem3.size(); i++) {
										Element elem4 = elem3.get(i);
										l.add(elem4.text());
									}
									kwds = StrUtils.join(StrUtils.LINE_REP, l);
								}
							}
						}

						{
							Elements elem2 = elem.get(0).getElementsByTag("pub-date");
							if (elem2.size() > 0) {
								Elements elem3 = elem2.get(0).getElementsByAttribute("pub-type");
								for (int i = 0; i < elem3.size(); i++) {
									Element elem4 = elem3.get(i);
									String val = elem4.attr("pub-type");
									String text = elem4.text();
									// if (val.equals("pmid")) {
									// pmid = text;
									// } else if (val.equals("pmc")) {
									// pmcid = text;
									// } else if (val.equals("doi")) {
									// doi = text;
									// }

									// System.out.println(val + "\t" + text);
								}
							}
						}

					}

					{
						Elements elem = doc.getElementsByTag("body");

						if (elem.size() > 0) {
							body = elem.get(0).text();
						}
					}

					{
						Elements elem = doc.getElementsByTag("ref-list");

						if (elem.size() > 0) {
							Elements elem2 = elem.get(0).getElementsByTag("ref");
							List<String> l = Generics.newArrayList();
							for (int i = 0; i < elem2.size(); i++) {
								Element elem3 = elem2.get(i);
								l.add(elem3.text());
							}
							refs = StrUtils.join(StrUtils.LINE_REP, l);
						}
					}

					// {
					// String[] tags = { "article-title", "abstract", "body", "kwd", "journal-title" };
					// for (int i = 0; i < tags.length; i++) {
					// Elements elem3 = doc.getElementsByTag(tags[i]);
					//
					// if (elem3.size() > 0) {
					// if (i == 0) {
					// title = elem3.get(0).text();
					// } else if (i == 1) {
					// abs = elem3.get(0).text();
					// } else if (i == 2) {
					// body = elem3.get(0).text();
					// } else if (i == 3) {
					// List<String> l = Generics.newArrayList();
					// for (int j = 0; j < elem3.size(); j++) {
					// l.add(elem3.get(j).text());
					// }
					// if (l.size() > 0) {
					// kwds = StrUtils.join(StrUtils.LINE_REP, l);
					// }
					// }
					// }
					// }
					// }

					List<String> vals = Generics.newLinkedList();
					vals.add(pmcid);
					vals.add(title);
					vals.add(abs);
					vals.add(body);
					vals.add(kwds);
					vals.add(jTitle);
					vals.add(pmid);
					vals.add(doi);

					for (int i = 0; i < vals.size(); i++) {
						vals.set(i, StrUtils.normalizeSpaces(vals.get(i)));
					}

					vals = StrUtils.wrap(vals);

					res.add(StrUtils.join("\t", vals));

					if (res.size() % batch_size == 0) {
						DecimalFormat df = new DecimalFormat("00000");
						String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(batch_cnt.getAndIncrement()));
						FileUtils.writeStringCollectionAsText(outFileName, res);
						res.clear();
					}
				}
			}
			tis.close();

			if (res.size() > 0) {
				DecimalFormat df = new DecimalFormat("00000");
				String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(batch_cnt.getAndIncrement()));
				FileUtils.writeStringCollectionAsText(outFileName, res);
				res.clear();
			}

			return (int) 0;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// {
		// TrecCdsDumper dh = new TrecCdsDumper(MIRPath.TREC_CDS_2014_COL_RAW_DIR, MIRPath.TREC_CDS_2014_COL_LINE_DIR);
		// dh.dump();
		// }

		{
			TrecCdsDumper dh = new TrecCdsDumper(MIRPath.TREC_CDS_2016_COL_RAW_DIR, MIRPath.TREC_CDS_2016_COL_LINE_DIR);
			dh.dump();
		}

		System.out.println("process ends.");
	}

	private Set<String> validDocIds;

	public TrecCdsDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		FileUtils.deleteFilesUnder(outPathName);

		List<File> files = Generics.newArrayList();

		for (File file : new File(inPathName).listFiles()) {
			if (file.getName().endsWith(".gz")) {
				files.add(file);
			}
		}

		int thread_size = files.size();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		batch_cnt = new AtomicInteger(0);

		batch_size = 2000;

		for (int i = 0; i < files.size(); i++) {
			String inFileName = files.get(i).getPath().replace("\\", "/");
			fs.add(tpe.submit(new Worker1(inFileName)));
		}

		for (int i = 0; i < fs.size(); i++) {
			fs.get(i).get();
		}

		tpe.shutdown();

		// makeSingleFile();
		// extractText();
	}

	// public void extractText() throws Exception {
	// System.out.printf("extract text from [%s].\n", outPathName);
	//
	// DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
	// dbf.setValidating(false);
	//
	// DocumentBuilder parser = dbf.newDocumentBuilder();
	//
	// parser.setEntityResolver(new EntityResolver() {
	//
	// @Override
	// public InputSource resolveEntity(String publicId, String systemId) throws SAXException, IOException {
	// if (systemId.contains("")) {
	// return new InputSource(new StringReader(""));
	// }
	// return null;
	// }
	// });
	//
	// TextFileReader reader = new TextFileReader(xmlFileName);
	// TextFileWriter writer = new TextFileWriter(outPathName);
	//
	// reader.setPrintNexts(false);
	//
	// while (reader.hasNext()) {
	// reader.printProgress();
	// String line = reader.next();
	//
	// String[] parts = line.split("\t");
	//
	// parts = StrUtils.unwrap(parts);
	//
	// String fileName = parts[0];
	// String xmlText = parts[1].replace(StrUtils.LINE_REP, "\n");
	//
	// // System.out.println(line);
	//
	// fileName = fileName.split("/")[2];
	//
	// String docId = FileUtils.removeExtension(fileName);
	//
	// if (validDocIds != null && !validDocIds.contains(docId)) {
	// continue;
	// }
	//
	// // xmlText = xmlText.replace("archivearticle.dtd",
	// // "F:/data/trec/cds/JATS-archivearticle1.dtd");
	//
	// Document xmlDoc = null;
	//
	// try {
	// xmlDoc = parser.parse(new InputSource(new StringReader(xmlText)));
	// } catch (Exception e) {
	// e.printStackTrace();
	// continue;
	// }
	// // org.w3c.dom.Document xmlDoc = parser.parse(docFile);
	//
	// String pmcId = "";
	// String title = "";
	// String abs = "";
	// String body = "";
	//
	// NodeList nodeList = xmlDoc.getElementsByTagName("article-id");
	//
	// for (int k = 0; k < nodeList.getLength(); k++) {
	// Element idElem = (Element) nodeList.item(k);
	// if (idElem.getAttribute("pub-id-type").equals("pmc")) {
	// pmcId = idElem.getTextContent().trim();
	// break;
	// }
	// }
	//
	// if (pmcId.length() == 0 || (validDocIds != null && !validDocIds.contains(pmcId))) {
	// continue;
	// }
	//
	// Element titleElem = (Element) xmlDoc.getElementsByTagName("article-title").item(0);
	// Element absElem = (Element) xmlDoc.getElementsByTagName("abstract").item(0);
	// Element bodyElem = (Element) xmlDoc.getElementsByTagName("body").item(0);
	//
	// if (titleElem != null) {
	// title = titleElem.getTextContent().trim();
	// }
	//
	// if (absElem != null) {
	// abs = absElem.getTextContent().trim();
	// }
	//
	// if (bodyElem != null) {
	// StringBuffer sb = new StringBuffer();
	// nodeList = bodyElem.getElementsByTagName("p");
	// for (int k = 0; k < nodeList.getLength(); k++) {
	// Element paraElem = (Element) nodeList.item(k);
	// String text = paraElem.getTextContent().trim();
	// text = text.replaceAll("[\\t]+", " ").trim();
	// sb.append(text + "\n");
	// }
	// body = sb.toString().trim().replace("\n", StrUtils.LINE_REP);
	// }
	//
	// if (pmcId.length() > 0) {
	// String[] values = new String[] { pmcId, title, abs, body };
	// for (int i = 0; i < values.length; i++) {
	// if (values[i].equals("null")) {
	// values[i] = "";
	// }
	// }
	// values = StrUtils.wrap(values);
	// String output = StrUtils.join("\t", values);
	// writer.write(output + "\n");
	// }
	// }
	// reader.printProgress();
	// reader.close();
	//
	// writer.close();
	//
	// }

	public void readValidDocIDs(String fileName) {
		validDocIds = new TreeSet<String>();
		TextFileReader reader = new TextFileReader(fileName);
		while (reader.hasNext()) {
			String line = reader.next();
			validDocIds.add(line.trim());
		}
		reader.close();
	}

}

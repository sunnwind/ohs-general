package ohs.ir.search.app;

import java.util.List;
import java.util.Map;

import org.apache.commons.compress.compressors.snappy.SnappyCompressorInputStream;

import ohs.corpus.type.RawDocumentCollection;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class WikiSynonymExtractor {

	public static void main(String[] args) throws Exception {
		WikiSynonymExtractor ext = new WikiSynonymExtractor(MIRPath.WIKI_COL_DC_DIR);
		ext.extract();
	}

	private RawDocumentCollection rdc;

	public WikiSynonymExtractor(String dir) throws Exception {
		rdc = new RawDocumentCollection(dir);
	}

	public void extract() throws Exception {

		List<String> lines = Generics.newLinkedList();

		for (int i = 0; i < rdc.size(); i++) {
			Map<String, String> m = rdc.getMap(i);
			String title = m.get("title");
			String content = m.get("content");

			List<String> sents = StrUtils.split("\n", content);
			
			
			

			if (sents.size() > 1) {

				String sent = sents.get(1);

				int idx = sent.indexOf("also known as");

				if (idx > 0) {
					String[] ps = { title, sent };
					ps = StrUtils.wrap(ps);
					lines.add(StrUtils.join("\t", ps));
				}
			}
		}

		FileUtils.writeStringCollectionAsText(MIRPath.WIKI_DIR + "fsents.txt", lines);

	}

	public void close() throws Exception {
		rdc.close();
	}

}

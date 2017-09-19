package ohs.nlp.pos;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

import ohs.io.TextFileReader;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.utils.Generics;

public class SejongReader implements Iterator<MDocument> {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		SejongReader r = new SejongReader(NLPPath.POS_DATA_FILE);
		while (r.hasNext()) {
			MDocument doc = r.next();
		}
		r.close();

		System.out.println("process ends.");
	}

	private TextFileReader reader;

	private MDocument doc;

	private int num_docs;

	private Set<String> posSet;

	public SejongReader(String dataFileName) throws Exception {
		reader = new TextFileReader(dataFileName);

		posSet = Generics.newHashSet();

		for (SJTag tag : SJTag.values()) {
			posSet.add(tag.toString());
		}

	}

	public void close() {
		reader.close();
	}

	private void filter(MDocument doc) {
		List<MSentence> sents = Generics.newArrayList();

		for (MSentence sent : doc) {
			boolean isValid = true;
			for (MToken tok : sent) {
				String pos = tok.getString(1);
				if (!posSet.contains(pos)) {
					isValid = false;
					break;
				}
			}

			if (isValid) {
				sents.add(sent);
			}
		}
		doc.clear();
		
		doc.addAll(sents);

		SejongParser.enumerateStarts(doc);
	}

	@Override
	public boolean hasNext() {
		StringBuffer sb = new StringBuffer();

		while (reader.hasNext()) {
			String line = reader.next();
			if (line.startsWith("</doc>")) {
				break;
			} else {
				if (!line.startsWith("<doc id")) {
					sb.append(line + "\n");
				}
			}
		}

		String text = sb.toString().trim();

		boolean hasNext = false;
		if (text.length() > 0) {
			doc = SejongParser.parseDocument(text);

			filter(doc);

			if (doc.size() > 0) {
				hasNext = true;
			}
			num_docs++;
		} else {
			doc = null;
		}

		return hasNext;
	}

	@Override
	public MDocument next() {
		return doc;
	}

}

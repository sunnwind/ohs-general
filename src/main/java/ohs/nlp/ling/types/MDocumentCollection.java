package ohs.nlp.ling.types;

import java.util.ArrayList;
import java.util.List;

import ohs.utils.Generics;

public class MDocumentCollection extends ArrayList<MDocument> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8530460881325620209L;

	public List<MSentence> getSentences() {
		int size = 0;
		for (MDocument d : this) {
			size += d.size();
		}
		List<MSentence> ret = Generics.newArrayList(size);

		for (MDocument d : this) {
			for (MSentence s : d) {
				ret.add(s);
			}
		}

		return ret;
	}

}

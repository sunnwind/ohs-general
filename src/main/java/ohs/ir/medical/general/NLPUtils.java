package ohs.ir.medical.general;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.TokenizerFactory;

public class NLPUtils {

	public static TokenizerFactory<? extends HasWord> tf = PTBTokenizer.factory(new CoreLabelTokenFactory(),
			"ptb3Escaping=false,normalizeParentheses=false,normalizeOtherBrackets=false");

	// ptb3Escaping=false,normalizeParentheses=false,normalizeOtherBrackets=false

	public static List<String> tokenize(String text) {
		DocumentPreprocessor documentPreprocessor = new DocumentPreprocessor(new StringReader(text));
		documentPreprocessor.setTokenizerFactory(tf);

		List<String> ret = new ArrayList<String>();

		for (List<HasWord> items : documentPreprocessor) {
			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < items.size(); i++) {
				sb.append(items.get(i).word());
				if (i != items.size() - 1) {
					sb.append(" ");
				}
			}
			ret.add(sb.toString());
		}
		return ret;
	}

}

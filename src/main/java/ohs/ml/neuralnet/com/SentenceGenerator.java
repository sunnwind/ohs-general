package ohs.ml.neuralnet.com;

import java.util.List;

import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.types.generic.Vocab;
import ohs.types.generic.Vocab.SYM;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class SentenceGenerator {

	private NeuralNet nn;

	private Vocab vocab;

	public SentenceGenerator(NeuralNet nn, Vocab vocab) {
		super();
		this.nn = nn;
		this.vocab = vocab;
	}

	public String generate(int max_words) {
		int w_unk = vocab.indexOf(SYM.UNK.getText());
		int w_start = vocab.indexOf(SYM.START.getText());
		int w_end = vocab.indexOf(SYM.END.getText());

		List<Integer> sent = Generics.newArrayList(max_words);
		sent.add(w_start);

		int w_prev = 0;

		while ((w_prev = sent.get(sent.size() - 1)) != w_end) {
			if (sent.size() - 1 == max_words) {
				sent.add(w_end);
				break;
			}
			DenseMatrix Yh = (DenseMatrix) nn.forward(new IntegerArray(sent));
			DenseVector yh = Yh.row(Yh.rowSize() - 1);

			VectorMath.cumulateAfterNormalize(yh, yh);

			int w = w_unk;

			do {
				w = ArrayMath.sample(yh.values());
			} while (w == w_unk || w == w_start);

			sent.add(w);
		}

		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < sent.size(); i++) {
			sb.append(vocab.getObject(sent.get(i)));
		}

		return sb.toString();
	}

}
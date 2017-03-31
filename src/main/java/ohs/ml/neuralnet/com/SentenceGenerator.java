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

	public List<String> generate(int max_words) {
		IntegerArray sent = new IntegerArray();
		sent.add(SYM.START.ordinal());

		while (sent.get(sent.size() - 1) != SYM.END.ordinal()) {
			if (sent.size() - 1 == max_words) {
				sent.add(SYM.END.ordinal());
				break;
			}
			DenseMatrix Yh = (DenseMatrix) nn.forward(sent);
			DenseVector yh = Yh.row(Yh.rowSize() - 1);

			VectorMath.cumulateAfterNormalize(yh, yh);

			int nw = SYM.UNK.ordinal();

			do {
				nw = ArrayMath.sample(yh.values());
			} while (nw == SYM.UNK.ordinal());

			sent.add(nw);
		}

		List<String> ret = Generics.newArrayList(sent.size());

		for (int i = 1; i < sent.size() - 1; i++) {
			ret.add(vocab.getObject(sent.get(i)));
		}

		return ret;
	}

}

package ohs.ir.medical.wiki;

import java.util.List;
import java.util.Set;

import ohs.io.TextFileReader;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class SentenceGenerator {

	public static void main(String[] args) {
		SentenceGenerator sg = new SentenceGenerator();
		sg.addTextLoc(1);
		// sg.setMaxTrainSents(10000);

		sg.process("../../data/medical_ir/wiki/wiki_medical_contents.txt.gz");

		Vocab vocab = sg.getVocab();

		long sum = vocab.sizeOfTokens();

		System.out.printf("vocab size:\t%d\n", vocab.size());
		System.out.printf("tokens:\t%d\n", vocab.sizeOfTokens());
		System.out.printf("sentences:\t%d\n", sg.getSentences().length);

	}

	private Set<Integer> textLocs;

	private Vocab vocab;

	private int cutoff_cnt;

	private int[][] sents;

	private int max_train_sents = Integer.MAX_VALUE;

	public SentenceGenerator() {
		textLocs = Generics.newHashSet();
	}

	public void addTextLoc(int loc) {
		textLocs.add(loc);
	}

	public int[][] getSentences() {
		return sents;
	}

	public Vocab getVocab() {
		return vocab;
	}

	public void process(String fileName) {
		int num_train_sents = 0;

		Set<Integer> locs = Generics.newHashSet();

		if (textLocs.size() > 0) {
			locs.addAll(textLocs);
		}

		vocab = new Vocab();

		{
			System.out.printf("read [%s]\n", fileName);
			Counter<String> wordCnts = Generics.newCounter();
			TextFileReader reader = new TextFileReader(fileName);
			reader.setPrintNexts(false);

			while (reader.hasNext()) {
				reader.printProgress();

				String[] parts = reader.next().split("\t");
				parts = StrUtils.unwrap(parts);

				if (locs.size() == 0) {
					for (int i = 0; i < parts.length; i++) {
						locs.add(i);
					}
				}

				for (int loc : locs) {
					String s = parts[loc].replace("<nl>", "\n");

					for (String sent : s.split("\n")) {
						sent = sent.toLowerCase();
						for (String word : sent.split(" ")) {
							wordCnts.incrementCount(word, 1);
						}
						num_train_sents++;

						if (max_train_sents == num_train_sents) {
							break;
						}
					}

					if (max_train_sents == num_train_sents) {
						break;
					}
				}

				if (max_train_sents == num_train_sents) {
					break;
				}
			}
			reader.printProgress();
			reader.close();

			wordCnts.pruneKeysBelowThreshold(cutoff_cnt);

			List<String> words = wordCnts.getSortedKeys();

			for (int i = 0; i < words.size(); i++) {
				vocab.add(words.get(i));
			}
		}

		{

			sents = new int[num_train_sents][];
			int[] word_cnts = new int[vocab.size()];
			int[] sent_freqs = new int[vocab.size()];

			TextFileReader reader = new TextFileReader(fileName);
			reader.setPrintNexts(false);

			int num_sents = 0;

			while (reader.hasNext()) {
				reader.printProgress();

				String[] parts = reader.next().split("\t");
				parts = StrUtils.unwrap(parts);

				for (int loc : locs) {
					String s = parts[loc].replace("<nl>", "\n");

					for (String sent : s.split("\n")) {
						sent = sent.toLowerCase();
						int[] ws = vocab.indexesOfKnown(sent.split(" "));
						sents[num_sents++] = ws;

						Set<Integer> wordSet = Generics.newHashSet();

						for (int w : ws) {
							word_cnts[w]++;
							wordSet.add(w);
						}

						for (int w : wordSet) {
							sent_freqs[w]++;
						}

						if (num_sents == num_train_sents) {
							break;
						}
					}
					if (num_sents == num_train_sents) {
						break;
					}
				}

				if (num_sents == num_train_sents) {
					break;
				}
			}
			reader.printProgress();
			reader.close();

			vocab.setWordCnts(new IntegerArray(word_cnts));
			vocab.setDocFreqs(new IntegerArray(sent_freqs));
			vocab.setDocCnt(num_sents);
		}

	}

	public void setMaxTrainSents(int max_train_sents) {
		this.max_train_sents = max_train_sents;
	}

}

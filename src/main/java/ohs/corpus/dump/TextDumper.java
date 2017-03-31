package ohs.corpus.dump;

import java.util.concurrent.atomic.AtomicInteger;

public abstract class TextDumper {

	protected String inPathName;

	protected String outPathName;

	public TextDumper(String inPathName, String outPathName) {
		this.inPathName = inPathName;
		this.outPathName = outPathName;
	}

	public abstract void dump() throws Exception;

	protected int batch_size = 20000;

	protected AtomicInteger batch_cnt = new AtomicInteger(0);

}

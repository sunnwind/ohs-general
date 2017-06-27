package ohs.ir.search.index;

import java.io.File;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DataCompression;
import ohs.corpus.type.DocumentCollection;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayUtils;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.ListMap;
import ohs.types.generic.ListMapMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * 
 * http://a07274.tistory.com/281
 * 
 * http://eincs.com/2009/08/java-nio-bytebuffer-channel/
 * 
 * @author ohs
 */
public class InvertedIndexCreatorOld {

	private int post_thread_size = 1;

	private int output_file_size = 5000;

	private int batch_size = 500;

	private DocumentCollection dc;

	private File dataDir;

	private File tmpDir;

	private int merge_thread_size = 1;

	private boolean encode = false;

	public InvertedIndexCreatorOld() {

	}

	public void create(String dataDir) throws Exception {
	}

	public void setBatchSize(int batch_size) {
		this.batch_size = batch_size;
	}

	public void setEncode(boolean encode) {
		this.encode = encode;
	}

	public void setMergingThreadSize(int thread_size) {
		this.merge_thread_size = thread_size;
	}

	public void setOutputFileSize(int output_file_size) {
		this.output_file_size = output_file_size;
	}

	public void setPostingThreadSize(int thread_size) {
		this.post_thread_size = thread_size;
	}

}

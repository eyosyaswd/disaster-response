	?p=
?cW@?p=
?cW@!?p=
?cW@	?]z?<???]z?<??!?]z?<??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?p=
?cW@?~j?t???A??ʡUW@YT㥛? ??*	      i@2U
Iterator::Model::ParallelMapV2ˡE?????!????eD@)ˡE?????1????eD@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateD?l?????!?jL?*A@)T㥛? ??1T?n?W?@:Preprocessing2F
Iterator::Model????????!쏗?(?H@);?O??n??1v??!@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!????%@);?O??n??1v??!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB`??"۹?!ph>?I@)????Mb??1aph>?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!HT?n?@)?~j?t?x?1HT?n?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!aph>???)????Mbp?1aph>???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?V-??!N+??d?A@)????Mb`?1aph>???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?]z?<??I|?a?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?~j?t????~j?t???!?~j?t???      ??!       "      ??!       *      ??!       2	??ʡUW@??ʡUW@!??ʡUW@:      ??!       B      ??!       J	T㥛? ??T㥛? ??!T㥛? ??R      ??!       Z	T㥛? ??T㥛? ??!T㥛? ??b      ??!       JCPU_ONLYY?]z?<??b q|?a?X@
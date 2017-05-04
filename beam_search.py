import theano
import numpy
import theano.tensor as T

theano.config.compute_test_value = 'warn'


# ......
#   ^
#   |
#  ...  ->  ...
#   ^
#   |
# ....
#   ^
#   |
#   .

print "building..."
wemb = theano.shared(numpy.random.randn(6, 4))
wr = theano.shared(numpy.random.randn(3, 5))
wf1 = theano.shared(numpy.random.randn(4, 3))
wr = theano.shared(numpy.random.randn(3, 3))
wf2 = theano.shared(numpy.random.randn(3, 6))

k = 3

# first step
init_h = T.vector('init_h')
init_h.tag.test_value = numpy.random.randn(3)
init_out = T.nnet.softmax(T.dot(init_h, wf2)).flatten()    # (6,)
init_idxout = init_out.argsort()[-k:]  # a vector of k indices
init_pout = init_out[init_idxout]      # a vector of k probabilities

init_xemb = wemb[init_idxout]
init_hid = T.dot(init_h, wr) + T.dot(init_xemb, wf1) # [k, 3] matrix with each line for hidden states of a squence

# htm1: hidden states of the previous state for each sequence
def step(htm1, pseq):
    out = T.nnet.softmax(T.dot(htm1, wf2))   # K * #vocab

    idxout = out.argsort(axis=1)[:, -k:]     # K * k
    pout = out[T.arange(k).dimshuffle(0, 'x'), idxout] # K * k
    pseq = pout * pseq.dimshuffle('x', 0)    # K * k
    selected_seq = pseq.flatten().argsort()[-k:] # k
    pseq = pseq.flatten()[selected_seq]      # k
    idxout = idxout.flatten()[selected_seq]  # k

    selected_idxtm1 = selected_seq // k      # k
    selected_htm1 = htm1[selected_idxtm1, :] # k * h

    xembt = wemb[idxout]                     # k * h
    ht = T.dot(selected_htm1, wr) + T.dot(xembt, wf1)

    return ht, pseq, idxout, selected_idxtm1

(h, seqprobs, idxt, selected_idx), updates = theano.scan(
    fn=step,
    outputs_info=[init_hid, init_pout, None, None],
    n_steps=10)

print "compiling..."
beam_search = theano.function([init_h],
                              [seqprobs, idxt, selected_idx, init_idxout])

print "running..."
seqprobs_np, idxt_np, selected_idx_np, idx0 = beam_search(numpy.random.randn(3))
invseq = []
last_idxpos = seqprobs_np[-1].argmax()
invseq.append(idxt_np[-1][last_idxpos])
for i in range(len(idxt_np)-1, 0, -1):
    prev_idxpos = selected_idx_np[i, last_idxpos]
    prev_idx = idxt_np[i-1, prev_idxpos]
    invseq.append(prev_idx)
    last_idxpos = prev_idxpos
invseq.append(idx0[selected_idx_np[0, last_idxpos]])



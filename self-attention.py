def self_attention( x, r = 8, name = 'sa', reuse = False ) :
    [ _, h, w, c ] = x.get_shape().as_list()
    n = h * w
    cr = c // r
    pf = conv2d( x, cr, 1, 1, name = name + '_f_1x1', reuse = reuse )
    pg = conv2d( x, cr, 1, 1, name = name + '_g_1x1', reuse = reuse )
    ph = conv2d( x, c, 1, 1, name = name + '_h_1x1', reuse = reuse )
    f = tf.reshape( pf, [ -1, n, cr ] ) 
    g = tf.reshape( tf.transpose( pg, [ 0, 3, 1, 2 ] ), [ -1, cr, n ] ) 
    h_ = tf.reshape( ph, [ -1, n, c ] ) 
    fg = tf.matmul( f, g ) # n by n matrix
    fg = tf.nn.softmax( fg, axis = 1 ) 
    att = tf.matmul( fg, h_ ) # n by c matrix
    att_fm = tf.reshape( att, [ -1, h, w, c ] ) 
    y = tf.get_variable( name + '_y', shape = [], dtype = tf.float32, initializer=tf.constant_initializer( 0 ) ) 
    return x + y * att_fm

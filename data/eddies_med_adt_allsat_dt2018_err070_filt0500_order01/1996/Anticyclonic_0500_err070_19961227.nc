CDF       
      obs    -   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ???t?j      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N#??   max       P?OH      ?  `   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?C?   max       =?      ?     effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>aG?z?   max       @ETz?G?       ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ?????R    max       @vi\(?       &?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @.         max       @P?           \  -?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?O        max       @?0           ?  .4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ????   max       >o??      ?  .?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A???   max       B*&?      ?  /?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A??A   max       B*=?      ?  0P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?C??   max       C??      ?  1   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P H   max       C??      ?  1?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  2l   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          [      ?  3    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          G      ?  3?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N"?   max       P??d      ?  4?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ???u%F   max       ??u??!?/      ?  5<   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?C?   max       =?      ?  5?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>aG?z?   max       @ETz?G?       6?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ?????R    max       @vg?
=p?       =?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @+         max       @P?           \  D?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?O        max       @??@          ?  E   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C
   max         C
      ?  E?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ??qu?!?S   max       ??u??!?/       Fx      %      %      )                           q                        i   	            #      1                     +      ?   '   )   \   =      N)??P ??N{?O?`?N%{gP?*N??_Nv?vN?ON҃?N??[O??UOU?P??P?OHO?@?N#??O-O2?NN???N???N???PC4DN?r?OV{O?59N??sO???NL?MPUN<??O??N=$?N???N?9Nc]?Os??O??SPL<?Oj?DO???Op&JO?$O6??N?fy?C??o<49X<49X<49X<D??<T??<e`B<u<?C?<?t?<???<?9X<?9X<ě?<?/<?/<??h<???=+=+=+=t?=?P=?P=?w=?w=?w=?w=#?
=<j=T??=ix?=y?#=y?#=}??=?o=?C?=??w=??
=?E?=??=??`=?G?=?a`]\`bnonnnooncbaaaa????
#/<SUNG</#?%!)257BJLB5)%%%%%%%%???
#/7@HJI</#
??????????????????????????????)5@JB*)???????????????????????\acgnvz??zna\\\\\\\\????????????????????ABOZ[chihhc[OLFDCBAA?????
?????????b\_mt?????????????}b??????????????????????
(>AC</
?????z???????)32????Uz???????????????????????


???????lhbmwz??????????zsml/--./<HUbnz}nlaH<8//0-06BOVXUOHB:6000000YRV[ghptz|yth[YYYYYY????????????????????????BJGCB>5)????????????????????????????????

?????>>MV]gt????????t[NB>????$???????;BO[ft???????th`WOB;????????????????????LLNV[t??????????t[SLNHOU[ahohf[ONNNNNNNN?????	
#)+,+&$#
??????????????????????@ABFN[eg`[NB@@@@@@@@gb_`gt?????tgggggggBBBO[dfg[OOBBBBBBBBB???????????????????????$)*'!?????tsz?????#$????{tHGE</##/4<CGH????????????????????????????

???????????

??????????????
" 
?????a_]^`annptz{?zngaa?_?l?x?????????????????x?l?j?_?W?_?_?_?_??"?;?N?[?a?^?N?6??	??????????????????????????????????z?u?{?????????y??????????y?m?`?T?G?=?>?<?B?N?`?m?s?y?ݽ???????????????ݽֽݽݽݽݽݽݽݽݿ`?y?????????????t?m?`?;?,?"??(?5?;?T?`?)?6?B?C?N?O?R?X?O?F?B?<?<?6?)?&?)?)?)?)?zÃÇÓßÓÏÇ?z?q?o?w?z?z?z?z?z?z?z?z???&?*?6?7?6?6?*???????????????????????u?r?g?f?Y?N?S?Y?f?r????????.?/?;?G?Q?J?G?;?.?"???"?-?.?.?.?.?.?.???????????????????????????s?f?]?O?[?s??????????"????????????????ݿܿ????)?5?B?P?S?I?<?1?)???????????????????)?-?g???????	?"?D?N?>?"?	???Z?D?1????-???????'?'????????߹ܹϹ̹Ϲܹ????@?M?N?Y?b?\?Y?W?M?@?4?3?4?>?@?@?@?@?@?@?[?h?t?yāĂććā?u?t?h?[?O?L?F?H?O?Z?[???????????????????????????????????????????????????????????????????!?-?:?>?F?G?F?E?:?-?!????!?!?!?!?!?!?#?/?<?????<?9?2?/?&?#?? ?"?#?#?#?#?#?#?#?<?L?U?`?_?a?U?I?0?
?????????????
??#?U?U?T?H?;?/?"????"?"?/?;?H?U?U?U?U?U?
??#?/?:?<?>?A?<?5?/?#?????
??
?
???(?5?A?M?T?Z?j?f?Z?N?A?(???????????????????????????????????????????????????????????????????????f?Z?S?T?Z?f?|????????????????????ùìéìðù???????????????"?3?;?E?E?C?=?/?"?	???????????????????ѿݿ??????????ݿҿѿͿпѿѿѿѿѿѿѿѾf?s??????ʾϾξ??????????s?f?c?O?K?W?f?????????????????????????????????????????tāĆčēĒčā?t?n?q?r?t?t?t?t?t?t?t?t?6?B?O?[?g?g?\?[?O?B?8?6?,?5?6?6?6?6?6?6???????????????ܻػܻܻ??????????????????????????????????????????}?}?????????????t¢?t?g?b?[?Q?L?N?[?t???ʼּݼ??????????ּʼ??????????????????o?b?V?J?J?V?b?o?{ǈǔǡǨǪǥǡǗǈ?{?o?zÇà??????????????????ùèàÓÇ?u?n?zD?D?D?D?D?D?D?D?D?D?D?D?D?D?D{DtDyD?D?D?E?E?E?E?E?E?E?E?E?E?EuEnEbEbEiE?E?E?E?E??n?{ŇōŐŎŊŇń?{?n?b?U?R?T?Y?^?b?i?n?????ûлܻ??߻ܻлû??????????????????? n B A V R Z ? A + B A G = d ~ 5 V  R *  l 1 i + t _ H V V C c 5 : 1 G : B ` H r K 4 G x    ~  d  ?  ?  Y  ?  ?  ?  ?  ?  ?  P  M    	?  (  V  l  ?  ?  ?  ?  X  !  -  ?  ?  J  ?    Q  z  N  ?  ?  ?  ?  9    ?  ?    P  ?  1????<??h<u=D??<T??=T??<?/<??
<?C?=?P<???=?P<?`B=H?9>C?=Y?=o=L??=q??=8Q?=<j=@?>bN=8Q?=P?`=T??=0 ?=??P=T??=?9X=P?`=?hs=y?#=?7L=??P=?C?=?"?=?E?>o??=???>?>E??>%?T>\)>VB'?BμBI B}'B";?B?(Bm?B??BZGB?BlaB1?BKB.OBD?B ??B$E?B 6?BQB=fB?HB?B?A???Bo?B	?.B??B^JB!??B
$B;B??B*&?Bb?B	??B?BڃB;?B?nB?B??B?RB??B\?B?`B'??B??BB)BC?B"?	BعB??B?BBh?BvB?MB9rBB
B??B??B ??B$:5B @NB?B??B?B?
B?\A??ABK B	??Bv?BK?B!?
B	?XBF#B??B*=?BC?B	??B??B?mBW?B8?BA?B??B?B?0B9KB?x@??}A??AF(Ai?lA-?8AjE?A??A?&NA???@?@Aa?PAH\A??A?WpA?5??C??@??A?q?AҨ?@Z?@undA?}?A???A??A??0A??AI?(AD??A?g?A?&?A|??AGkqA"rNA?Z?A?=R@???A???A???@?Y?B@qA͢?C??C??A??@?i?@??!A??AE?Aj?fA.?RAj?KA?x?AȝA???@?˦Aa?NAF??A?L?A???A?N ?P H@?bwA?$?A??@[??@t	A?}?A??A?kbA??A??FAI ?AD??A?r#A?n"A|?<AB??A!5?Aݒ?A??$@??<A?jA???@??+B??A??&C??PC??A? @?(?      &      &      )               	            q                        j   	            $      1                     ,      ?   '   )   ]   =            '            +                  %      +   [                        +         #            +                           /                                    '                  #         G                                 #            %                           %                  N)??O?PN{?Ou?bN%{gO???N??_Nv?vN?ON??N??[O??`OU?O^SP??dO?@?N#??O??N?NZ?N???N{?O???N?r?N???O?59N"?O;fNL?MO??>N<??O??N=$?N???N?9Nc]?Oc}?O??SP??Oj?DOI??Op&JON\-O6??N?fy    ?  u  ?  3  c  ?  ?  ?  ;  ?  #  C  ?  
:  ?  ?  ?  ?  j  ?  K  
?  ?  ?  ?  "  3    b  ?  ?  ?  ?  ?  ?  	`    @  ?  ?  _  ?  	?  ??C?;??
<49X<?o<49X<???<T??<e`B<u<?1<?t?<?1<?9X=C?=49X<?/<?/<???=??=?P=+=C?=???=?P=??=?w=#?
=<j=?w=H?9=<j=T??=ix?=y?#=y?#=}??=?+=?C?=??=??
=\=??=??=?G?=?a`]\`bnonnnooncbaaaa#/8<EFA</#
%!)257BJLB5)%%%%%%%%????
#/4<>DFD</#
???????????????????????????8<>C5)???????????????????????\acgnvz??zna\\\\\\\\????????????????????IFFIOR[`ge_[QOIIIIII?????
?????????f`brt?????????????sf???????????????????????

??????????????),????????????????????????????


???????mmmz|?????????zunmmm3015<GHUU^XUH<3333334166BOOSOOCBA6444444YRV[ghptz|yth[YYYYYY?????????????????????????)/4543) ???????????????????????????

??????>>MV]gt????????t[NB>?????????????XU^hnt?????????thg]X????????????????????RRSXgt??????????{t[RNHOU[ahohf[ONNNNNNNN?????	
#)+,+&$#
??????????????????????@ABFN[eg`[NB@@@@@@@@gb_`gt?????tgggggggBBBO[dfg[OOBBBBBBBBB??????????? ?????????????$)*'!???????????????????HGE</##/4<CGH????????????????????????????

?????????????

??????????
" 
?????a_]^`annptz{?zngaa?_?l?x?????????????????x?l?j?_?W?_?_?_?_??"?/?;?H?O?N???/?%??	???????????????????????????????z?u?{?????????m?y?????????}?y?p?m?`?T?G?B?C?B?H?T?e?m?ݽ???????????????ݽֽݽݽݽݽݽݽݽݿy?????????????w?`?T?G?;?/?,?.?2?;?O?`?y?)?6?B?C?N?O?R?X?O?F?B?<?<?6?)?&?)?)?)?)?zÃÇÓßÓÏÇ?z?q?o?w?z?z?z?z?z?z?z?z???&?*?6?7?6?6?*????????????f?r???????????r?f?Y?T?Y?[?f?f?f?f?f?f?.?/?;?G?Q?J?G?;?.?"???"?-?.?.?.?.?.?.??????????????????????????s?U?_?f?u???????????"????????????????ݿܿ??????)?5?B?G?;?5?+???????????????
??5?Z??????????<?D???/?	?????g?S?J?<?)?5???????'?'????????߹ܹϹ̹Ϲܹ????@?M?N?Y?b?\?Y?W?M?@?4?3?4?>?@?@?@?@?@?@?h?t?t?āąĄā?t?h?[?O?N?H?I?O?[?e?h?h???????????????????????????????????????????????????????????????????!?-?:?>?F?G?F?E?:?-?!????!?!?!?!?!?!?#?/?<???>?<?8?1?/?(?#? ?"?#?#?#?#?#?#?#?
??#?0?=?J?N?M?F?<?0?#?
?????????????
?U?U?T?H?;?/?"????"?"?/?;?H?U?U?U?U?U??#?/?9?<?=?@?<?4?/?#????????????(?5?A?M?T?Z?j?f?Z?N?A?(??????????????????????????????????????????????????s???????????????????s?m?f?`?X?Y?Z?f?s????????????????ùìéìðù????????????????"?/?7?;?9?0??	?????????????????????ѿݿ??????????ݿҿѿͿпѿѿѿѿѿѿѿѾf?s??????ʾϾξ??????????s?f?c?O?K?W?f?????????????????????????????????????????tāĆčēĒčā?t?n?q?r?t?t?t?t?t?t?t?t?6?B?O?[?g?g?\?[?O?B?8?6?,?5?6?6?6?6?6?6???????????????ܻػܻܻ????????????????????????????????????????????~?~???????????t¢?t?g?b?[?Q?L?N?[?t???ʼּ??????????????ּʼ????????????????o?b?V?J?J?V?b?o?{ǈǔǡǨǪǥǡǗǈ?{?où??????????????????ùìàÓÅÇÓàìùD?D?D?D?D?D?D?D?D?D?D?D?D?D?D{DtDyD?D?D?EuE?E?E?E?E?E?E?E?E?E?E?E?E?EuEsEmEiErEu?n?{ŇōŐŎŊŇń?{?n?b?U?R?T?Y?^?b?i?n?????ûлܻ??߻ܻлû??????????????????? n ; A R R Y ? A + < A E = K x 5 V  , .  m ( i , t N 3 V N C c 5 : 1 G 8 B V H Z K  G x    ~    ?  "  Y  V  ?  ?  ?  ?  ?  ?  M  ?  ?  (  V  7  ?  l  ?  ?  ?  !    ?  \  ?  ?  9  Q  z  N  ?  ?  ?  ?  9  ?  ?  ?    ?  ?  1  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
  C
          !  '  -  ,  '  "                ?   ?   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  W  +    ?  ?  ]    ?  ?  u  x  {  ~  ?  ?    |  y  v  l  ^  O  @  1        ?   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  H    ?  n    ?  K  ?  ?  ?   ?  3  .  )  $               ?   ?   ?   ?   ?   ?   ?   ?   ?   ?    ;  T  b  ]  J  .      ?    4  I  >    ?  D  ?  2  D  ?    2  F  L  C  ?  f  M  0    ?  ?  T  
  ?  q  "  ?  ?  ?  ?  ?  }  w  r  l  e  ^  V  S  S  T  V  \  c  i  n  s  x  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  w  r  m  h  #  &  .  8  9  .    ?  ?  ?  ?  {  S  %  ?  ?  ?  d  &  ?  ?  ?  ?  ~  {  w  n  e  W  H  4      ?  ?  ?  ~  \  :        "         ?  ?  ?  [    ?  ?  ?  ?  ?  ?  s  W  =  C  5  '      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  d  4     ?  ?  ?  ?  ?  u  n  o  ?  ?  ?  ?  s  F    ?  {    ?  Y   ?  	g  	?  	?  
#  
,  
  	?  	E  ?  ?  F  ?  ?  	  ?     R  Z    ?  ?  r  G  )    ?  ?  ?  ?  ?      ?  ?  ?  z  M     ?  ?  ?  8  0  (  !          (  5  B  C  <  5  .  %      	  k  ?  ?  ?  ?  t  b  N  8    ?  ?  ?  G  ?  ?  =  ?  u    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  M    ?  ?  /  ?  ?  8  A  I  R  Y  a  g  d  [  E  +    ?  ?  ?  m  C    ?  ?  ?  ?  ?  ?  ?  ?  ?  u  d  P  6    ?  ?  ?  ?  _  1    ?  A  I  K  G  =  *    ?  ?  c    ?  ?  H  ?  ?  g    ?  ?  	   	?  	?  
M  
?  
?  
?  
?  
?  
?  
?  
?  
B  	?  	'  j  ?  p  ?  =  ?  ?  u  b  N  8      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  z  ?  ?  {  n  _  L  6    ?  ?  ?  ?  I    ?  c    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  ]  =    ?  ?  w  D     ?  ?  y          !  #  %  &  (  *  *  (  '  %  #             {    )  1  3  0  (      ?  ?  ?  R    ?  K  ?  @  h   ?      $  +  1  6  9  5  ,    
  ?  ?  ?  ?  b  :    ?  ?  ?  )  I  ]  `  Z  U  J  5      ?  f  +  ?  ?  l  ?  ?  6  ?  4  *        ?  ?  ?  ?  ?  ?  ?    f  M  0     ?   ?  ?  ?  ?  ?    ]  9    ?  ?  ?  X    ?    ?  ?  j  ?  1  ?  ?  z  p  f  [  M  ?  1  #       ?   ?   ?   ?   ?   ?   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  j  U  @  +      ?  ?  ?  ?  ?  ?  ?  y  ]  ;    ?  ?  ?  h  4  ?  ?  ?  Z  #  ?  ?  ?  ?  ?  w  j  ^  P  C  7  +      ?  ?  ]  <    ?  	Y  	]  	N  	G  	D  	'  	  	  ?  ?  d    ?  a  ?  ?  9  ~  "  ?      ?  ?  ?  ?  ?  ?  ?  s  Q  %  ?  ?  ?  M  /  ?  Z  \  ?  ?  	  <  <  0  +  .  +    ?  ?    y  ?  ?  ?  
    +  ?  }  h  O  3    ?  ?  ?  ]    ?  {    ?  W  ?  Z  T    ?  ?  ?  ?  ?  ?  ?  {  h  R  !  ?  ?  K  ?  8  ?  ?  ?    _  S  N  F  F  >  -  	  ?  ?  S  ?  u  ?  ?    ?  
2  2  ?  <  '  #  h  ?  ?  ?  ^  )  
?  
?  
?  	?  	*  W  b  `  X  L  ?  	?  	p  	>  	  ?  ?  ?  H    ?  ?  -  ?  p     q  ?  $  u     ?  ?  q  ;    ?  ?  ?  ?  ?  f  "  ?  j    ?  3  ?  S  ?
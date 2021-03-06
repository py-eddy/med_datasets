CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ????vȴ       ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N?   max       P???       ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?C?   max       =ȴ9       ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?33333   max       @F\(??     
P   ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @va\(?     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @*         max       @N?           ?  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?        max       @??`           5?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??`B   max       >??       6?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A???   max       B4Z?       7?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?c?   max       B4J?       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ??T?   max       C??       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ???    max       C??       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N?   max       P4?a       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??333333   max       ???????       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?C?   max       >?       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>??????   max       @F\(??     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @va\(?     
P  L?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @#         max       @N?           ?  V?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?        max       @?0            Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Cd   max         Cd       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???????A   max       ??	k??~(     ?  Y|               B                        ?                              	         
            L   N                           ?      	   	         +         G               K                        8            	Nx7O?E?O?_N/??P???NQ?OU??N??N??]N]?0N?#O-?P???O?+N?'?N!?O ?O???Nfn?O?5?N?N%??N???N?\mO y\O??]OON??uOF??P81/O?? N?(?NeߛO3GiN2??O?	%N??ON?S?N??kP?tN?fhN??N???O:9cOp,O??N?`sN?)?P>??N???O2FOx??Nn?O?GOC?<N?N???NF\WNT
?O?-Ov?	O?x?N?_CN$?MN?@nN???C??49X?#?
?D???D??:?o:?o:?o:?o;o;o;?o;?`B;?`B<t?<49X<u<?t?<???<??
<??
<?1<?9X<ě?<ě?<ě?<???<?/<?`B<?`B<??h<?<?<???<???<???=o=\)=\)=?P=?P=??=??=?w=?w=#?
=#?
='??='??='??=,1=<j=D??=H?9=H?9=P?`=Y?=Y?=]/=m?h=?7L=?C?=?C?=?O?=?9X=ȴ9?????????????????????8669BN[gt???wtg[NB?????????????????????#&/572/#,,>A>@GWh??????tOB6, ?	"/7;@B:0/"	????????????????????????????????????????)56<5-)'??????????????????????????????
?????vz}?????'-,$??????v#/7<KSZ\ZUH/# ????????????????????ONBABBLOQ[[[YOOOOOOO??????????????????????????
!&('$&
????)*)(!t?????????????????zt**6BO[ahhh[OB6******GHIU[ahbaUQHGGGGGGGG????????????????????????)*)????=;BN[gtuutig[RNFB??????????		?????????????	?????INS[bgt?????ttgd[YNI????????????????????)H[ght?tu`b]OB( ??????????????????????????????????????????????????????????940<=HUVamnoqwnaUH<9419<EHUaUHE<44444444??????????"))%" #(*/111/'#'((+-/1<DGD><4//''''<NG'????????????)6<geeght|???????ztpmhg@98BN[_grtytjge[NB@@~y|???????????~~~~~~bfmz?????????????znbst???????????????us????????????????????????????!",/2;HLNHH;/"????%)5>LOLB5)??
	
#0960##




?	
#0:<B?<;50#
?ttvvz?????????????zt???????????????????????????????????????
??????B?>BOQOKOPOBBBBBBBBB$#)55BNWQNHBA75)$$:79<HQUNH<::::::::::iqt??????tiiiiiiiiiia\[amz}????{zmhca__a???????????????????????????????????????????????????????????


???????????#,/&# 

#######
	
?????????????????????????????????H?U?a?n?zÅÇÆÉÇÂ?z?n?U?Q?F?@?@?F?H???????????????????????????????y?x?m?y????????????????????????????????????????????4?Z?s?????׾??????׾????s?;?4?
? ??????????????üùöù?????????????????????H?T?a?m?v?z?|?????z?u?m?a?T?H?F?=?>?G?H?)?6?B?;?6?2?)?"????
????(?)?)?)?)?y???????????????y?v?n?s?y?y?y?y?y?y?y?y?Z?f?k?i?g?f?Z?P?M?K?M?R?Z?Z?Z?Z?Z?Z?Z?Z?#?/?:?<???<?0?/?.?&?#??#?#?#?#?#?#?#?#?	??"?.?/?7?;?=?;?9?/?"???	???????	Ó??????????????????ùàÓ?[?C?M?O?a?zÓ?`?m?y?~?????y?m?T?G?;?.?"?????.?G?`???'?3?:?=?9?3?'?????	???????l?l?r?y?????????????????y?l?l?l?l?l?l?l???????????????????????????????ſ???????)?B?P?P?@?5?)????????????????????ݿ???????
???????????ݿڿݿݿݿݿݿݾ???????????????????s?s?h?e?f?s?????????#?)?-?+?'?#?????
?
?????????'?)?.?)?"?????????????????ʼּܼۼۼּҼʼǼ????????????????????????????????????????????????????????????m?y???????????????y?w?m?j?e?`?\?`?e?m?m?	????"?(?$? ??	???????־ؾ޾??????	?S?_?l?x?{?????~?x?l?_?T?S?F?@?A?F?M?S?S??????	???????????????ݽݽܽݽ??????(?4?M?Z?_?d?b?Z?M?A?4?(???????л???4?@?M?T?U?4????û????????x????????5?N?Y?e?d?N?C?E?A?5?(?#???
??
???????????????????????????????????????????????????????????????????????????????????zÇÓÞàëíìåàÞÓÇ?z?t?r?o?r?w?z?????????????????????????????????????	??.?;?T?a?e?X?T?G?.?"??	???????????	???????????????????????????????????????????(?5?8?A?N?Z?N?A?3?(???
??????Z?g?s?????????????????s?o?g?Z?Y?Z?Z?Z?Z???ּ????y?p?r?y?f?[?r?????ּ??????????Ｄ?????ʼʼͼʼȼ???????????????????????O?[?h?k?l?k?h?b?[?U?O?J?B?B?B?E?I?N?O?O?)?6?B?G?O?Q?O?H?B?6?-?)?&?'?)?)?)?)?)?)?tāčėěĜĚĔčĆā?y?t?p?l?h?d?b?h?t?ʾ׾??????????????????׾ϾѾѾ˾þ??Ǿ????????????????????????????????????????????	???"?-?/?1?/?#?"??	??????????????#?0?<???I?K?I?B?D?<?0?#?#????#?#?#?#?#?<?I?U?uŇœ?b?U?<?0?
??????????????#?f?s???????????????|?s?s?f?d?e?f?f?f?f??????
???????????????ݼ׼ݼ?????????)?A?=?<?9?6?+?)?????????????????????????????????????????????????????????????$?$????????²«¤¥¢§¿?????y?????????????????}?y?s?l?k?e?b?d?f?l?y????????????????????׺ߺ????????????????????	???"?"?%?'?"???	??????????????D?D?EE	EED?D?D?D?D?D?D?D?D?D?D?D?D?D??/?;?>?E???;?/?&?'?'?/?/?/?/?/?/?/?/?/?/??????????????????ŹŭŠşśŠŭŴŹ???????????????????????????????????????????????????????????????????s?g?N?A?9?L?g?s?????(?5?7?5?0?*?(???
????	????E?E?E?E?E?E?E?E?EvE|E?E?E?E?E?E?E?E?E?E?ǭǡǔǈǇǈǌǔǖǡǭǳǳǭǭǭǭǭǭǭ?H?<?0?#?????#?0?:?<?I?J?H?H?H?H?H?H S 1  I V T ! N & O V ' * Q 2 . Q = W ' l B . I C R  ? F g H M C 0 f E g X z U 7 ^ ! C r  * 0 Y ` ) , O @ e s Q " X V B s , l H p  ?    D  Y  ?  M  ?  ?  ?  ?  V  s  ;  ?  ?  A  <  ?  ?  L  B  G  ?    8  f  I  +  ?  ?      ?  ?  ?  c  ?  >    ?    $  ?  ?  ?  N      ?  ?    ?  h  H  ?    ,  Y  ?  y  	  ?  ?  ?  ?  ???`B<?C?<e`B:?o=}??<u<???<e`B;?o;?`B;?`B<?9X>V=+<??
<?9X<???=D??<ě?=t?<?`B<?`B<?<???=\)=+=@?<?=q??=???=??=?P=t?=49X=0 ?=aG?=\)=T??=<j>??=]/=@?=<j=m?h=@?=???=m?h=<j=?`B=<j=?o=???=aG?=???=??=e`B=?7L=?+=q??=?\)=?j=???=??
=??
=???=?"?B!?B?-B"VB??B??B+\A?%B??B,?FB?"B?0B?GBA?BmsB?LB?pB??Bj?B"?B{?Bs0B?NB"V9B??B?AB"~XBh?B	BBo?BB4B9?B??B??B?DB%B??A???B??B]>BC?BtgB??B
?B Y?B4Z?B4cB??A???BKBB%	?B%.?BP=B!??B??B,??B?#B`B?B
?A??QBBF?B$?Bc?BNoB&?B!?OB?hB"G8B=3B?BA?A?p?B?;B,?~B??B?RB?lB:?B?gB?B?KB??B??B>?BEmB?9B?bB"Y?B@ B?wB"?uB?KB	?MB??B??B:?B?XBxhB?vB??B?oA?c?B?bB?B@BP?B? B
?-B E?B4J?B0?B)?A???B??B$?WB%?B??B!??B?B,?BA?B4WB7gB
A?A??B??B??B??B?BC?BB%@???A?0@???A???AG??A??YA??tA?DAn?NA???A?,?A??A?>NAe?z??T?AlA?eA??QA?
AH7?A??#A???@?^?At?;Am;MAY?/@???A/`"A:	-@?ٙA?|yA??]B?A?ԲA?(UA_?<A??A???A?@?]?@??A?0?A???A?W?ARZA?? A?3EA?p?A??ADG'AeA? _A!??A?}yA!P@L??A???C?J A???A???A??A?)?A??+C??B?_A?sX@??&A??@?|oA??NAJ?/A?y?A??Aե-Anb?A>?A?A?b?A?c?Ad????? A?A??WA???A??QAH?A???A??)@?wAs?Am3?AZ?z@?PA0??A<0?@??KA??'A???B??A??>A?~?A`?A??UA??hA?p?@?N@?HA?A???A݂?AU?A?NA?dtA?\?A?|?AC??AF?AԬ?A!?A??$A??@S?A?w#C?IA?TIA?I?A???A?q*A?-SC??B??A?d               B                        ?                        	      	         
             L   N                           ?      
   	      	   +         H               L                        9            
               ;                        =               !                                    9   !                           :                           3               %                        #                           #                                       !                                                                                             /                                                   Nx7O?E?N?]kN/??Oǎ<NQ?N??N??N??]N]?0N?#OӞO??O?.N?'?N!?NaK8O??Nfn?Ox4N?N%??N???N?\mN?p?O??]N?P_N??uO*?N?N?O[ ?N?(?NeߛO3GiN2??O??-N??ON?S?N??kO??N?ilN??N???O:9cOp,O??wN?91N?)?P4?aN???O2FOx??Nn?O???N?? N?N???NF\WNT
?O?-Ov?	O$?N?_CN$?MN?@nN??  ?  ?  ?  ?  ?  L    ?  w  .  {    
?  ?  ?  d  ?  ~  1      ?  B  ?  ?  6  ?  {  ?  ?  
?  ?  ?  ?  $    ?  K  ?  ?    D  M  ?  X  ?  ?  ?  
h  ?  ?  ?  ?  
?  ?  Q  ?    ?      
p  ?     ?  y?C??49X?o?D??<???:?o<D??:?o:?o;o;o;ě?=?1<o<t?<49X<???<?1<???<?9X<??
<?1<?9X<ě?<???<ě?<??h<?/<???=??w=T??<?<?<???<???=+=o=\)=\)>?='??=??=??=?w=?w=49X='??='??=49X='??=,1=<j=D??=y?#=aG?=P?`=Y?=Y?=]/=m?h=?7L=?1=?C?=?O?=?9X=ȴ9?????????????????????8669BN[gt???wtg[NB?????????????????????#&/572/#MMNS[ht???????th[XVM	"/46/-"	????????????????????????????????????????)56<5-)'??????????????????????????

????????????????????????????#/5<JRY[YUH/#!????????????????????ONBABBLOQ[[[YOOOOOOO???????????????????????????
!#!#
????)*)(!~x?????????????????~**6BO[ahhh[OB6******GHIU[ahbaUQHGGGGGGGG????????????????????????)*)???A?>BN[gtttthg[PNJBAA????????		??????????????????INS[bgt?????ttgd[YNI????????????????????53356>BOQVSQOHB65555??????????????????????????????????????????????????????????940<=HUVamnoqwnaUH<9419<EHUaUHE<44444444??????????"))%" #(*/111/'#'((+-/1<DGD><4//''''??????? $"?????hhjt??????tqjhhhhhhh@98BN[_grtytjge[NB@@~y|???????????~~~~~~bfmz?????????????znbst???????????????us?????????????????????????
???!",/2;HLNHH;/"????#'5<JNJB5)??
	
#0960##




?	
#0:<B?<;50#
?ttvvz?????????????zt???????????????????????????????????????????????????B?>BOQOKOPOBBBBBBBBB$#)55BNWQNHBA75)$$:79<HQUNH<::::::::::iqt??????tiiiiiiiiiia\[amz}????{zmhca__a???????????????????????????????????????????????????????????


???????????#,/&# 

#######
	
?????????????????????????????????H?U?a?n?zÅÇÆÉÇÂ?z?n?U?Q?F?@?@?F?H?????????????????????????{???????????????????????????????????????????????????????????????;Ӿо???????s?f?\?S?M?R?Z?f??????????????üùöù?????????????????????a?c?m?p?u?r?m?a?W?T?L?K?T?Z?a?a?a?a?a?a?)?6?B?;?6?2?)?"????
????(?)?)?)?)?y???????????????y?v?n?s?y?y?y?y?y?y?y?y?Z?f?k?i?g?f?Z?P?M?K?M?R?Z?Z?Z?Z?Z?Z?Z?Z?#?/?:?<???<?0?/?.?&?#??#?#?#?#?#?#?#?#?"?)?/?4?;?7?/?&?"???	?? ??	???"?"ÇÓàìù??????ùöìàÓÇ?{?u?r?s?zÇ?`?m?y?}?????y?m?T?G?;?.?"????"?.?G?`???'?3?:?=?9?3?'?????	???????l?l?r?y?????????????????y?l?l?l?l?l?l?l???????????????????????????????????????????)?B?I?M?D?<?5?)???????????????????ݿ???????
???????????ݿڿݿݿݿݿݿݾ???????????????????????}?s?l?i?s?x?????#?)?-?+?'?#?????
?
?????????'?)?.?)?"?????????????????ʼּܼۼۼּҼʼǼ????????????????????????????????????????????????????????????m?y???????????????y?x?m?l?f?`?^?`?j?m?m?	????"?(?$? ??	???????־ؾ޾??????	?S?_?l?x?y????x?t?l?a?_?S?G?F?E?F?Q?S?S??????	???????????????ݽݽܽݽ????4?M?Z?]?`?b?_?Z?M?A?.?(???????(?4?лܻ??????????????ܻٻлʻλллл???(?5?A?M?N?Z?[?T?N?A?5?(?"???????????????????????????????????????????????????????????????????????????????????????zÇÓÞàëíìåàÞÓÇ?z?t?r?o?r?w?z?????????????????????????????????????	?"?.?;?G?]?b?V?T?G?.?"??	??????????	???????????????????????????????????????????(?5?8?A?N?Z?N?A?3?(???
??????Z?g?s?????????????????s?o?g?Z?Y?Z?Z?Z?Z???????ʼݼ??????ּʼ??????????????????????????Ǽ¼??????????????????????????????O?[?h?k?l?k?h?b?[?U?O?J?B?B?B?E?I?N?O?O?)?6?B?G?O?Q?O?H?B?6?-?)?&?'?)?)?)?)?)?)?tāčėěĜĚĔčĆā?y?t?p?l?h?d?b?h?t?ʾ׾??????????????????׾ϾѾѾ˾þ??Ǿ????????????????????????????????????????????	???"?,?/?1?/?"?"??	??????????????#?0?<???I?K?I?B?D?<?0?#?#????#?#?#?#?#?<?I?U?r?}ł?b?U?:?0?
??????????????#?f?s???????????????|?s?s?f?d?e?f?f?f?f??????
???????????????ݼ׼ݼ?????????)?A?=?<?9?6?+?)?????????????????????????????????????????????????????????????
?????
??????¿²°±¬´¿?ؽ????????????????~?y?t?l?j?l?o?p?y??????????????????????????׺ߺ????????????????????	???"?"?%?'?"???	??????????????D?D?EE	EED?D?D?D?D?D?D?D?D?D?D?D?D?D??/?;?>?E???;?/?&?'?'?/?/?/?/?/?/?/?/?/?/??????????????????ŹŭŠşśŠŭŴŹ?????????????????????????????????????????????s?????????????????????????????y?s?g?g?s???(?5?7?5?0?*?(???
????	????E?E?E?E?E?E?E?E?EvE|E?E?E?E?E?E?E?E?E?E?ǭǡǔǈǇǈǌǔǖǡǭǳǳǭǭǭǭǭǭǭ?H?<?0?#?????#?0?:?<?I?J?H?H?H?H?H?H S 1  I 5 T 5 N & O V    S 2 . : > W  l B . I F R  ? ? @  M C 0 f B g X z  1 ^ ! C r  % 0 Y ` ) , O < J s Q " X V B X , l H p  ?    ?  Y  ?  M  ?  ?  ?  ?  V  :  <  ?  ?  A  l  ?  ?  ?  B  G  ?    !  f    +  ?  ?  ?    ?  ?  ?    ?  >    y  ?  $  ?  ?  ?    ?    ?  ?    ?  h  ?  ?    ,  Y  ?  y  	  }  ?  ?  ?  ?  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  Cd  ?  ?  ?  ?  ?  ?  ?  ?  ?  k  o  ?  ?  ?  ?  ?  ?  ~  k  X  ?  ?  ?  ?  ?  v  R  (  ?  ?  ?  P  
  ?  W  ?  ?  R  
  ?  X  v  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  Z  ,  ?  ?  V  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  k  ?  ?  ?  2  u  ?  ?  ?  ?  ?  ?  \    ?  w    }  ?  ?  L    =  r  f  m  y  x  :  $    ?  ?  s  @    ?  ?  h  0  ?  )  g  ?  ?  ?  ?        ?  ?  ?  ?  V  ?  Y  ?  &    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  f  @    ?  ?  ?  l  w  o  h  `  X  P  I  >  3  '         ?   ?   ?   ?   ?   ?   ?  .  -  ,  *  '        ?  ?  ?  ?  x  P  )     ?   ?   ?   o  {  w  r  m  h  b  \  V  M  >  0  !    ?  ?  ?  ?  ?  q  W    
      
    ?  ?  ?  ?  ?  ?  ?  u  T  2    ?  ?  ?  ?  }  ?  I  ?  ?  	c  	?  
9  
?  
?  
?  
?  
?  
$  	t  P     _  9  ?  ?  ?  ?  ?  ?  z  b  H  1    ?  ?  ?    @  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  g  O  5    ?  ?  ?  ?  ?  o  D    ?  ?  d  I  /      ?  ?  ?  ?  }  `  B  $    ?  ?  ?  ?  ^  :  p  p  p  r  t  x  ~  ?  ?  r  \  D  *    ?  ?  ?  ?  ?  j  W  q  }  z  q  b  L  .  
  ?  ?  ?  ?  O    ?  ?  .  ?   ?  1  *  #        ?  ?  ?  ?  ?  ?  ?  y  `  G  ,     ?   ?  ?  	          ?  ?  ?  ?  ?  ?  ~  V  !  ?  ?  ?  w  ?             ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  e  Q  ?  ?  ?  ?  ?  ?  ?  ?  m  U  =  $    ?  ?  ?  ?  r  R  2  B  <  7  /  &      ?  ?  ?  ?  ?  ?  z  ^  K  ;  ,      ?  ?  ?  ?  ?  ?  {  p  f  Y  K  =  ,    
  ?  ?  ?  ?  s  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  d  O  7    ?  ?  ?  b    ?  6  -  $      ?  ?  ?  ?  ?  ?  ?  ?  q  ^  I  1     ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  j  L  (    ?  ?  L  ?  ?  /  ?  ?  {  w  s  p  l  h  d  \  R  G  =  3  (         ?   ?   ?   ?  ?  ?  ?  ?  ?  ?  ?  x  Y  D  *    ?  ?  Z    ?  v  2  ?  ?  ?  ?  ?  ?      +  !    -  ?  ?  ?  ?      ?      ?  	u  
A  
s  
?  
?  
?  
?  
?  
e  
4  	?  	?  	  ?  ?    )  ?  ?  ?  ?  ?  ?  v  k  _  P  A  -      ?  ?  ?  ?  ?  ?  h  M  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  i  X  F  2    ?  ?  ?  ?  ?  ?  ?  ?  ^  G  +        ?  ?  ?  ?  ?  p  M  4        $  W  m  X  C  /      ?  ?  ?  ?  ?  y  b  ]  _  d  l  w                ?  ?  ?  ?  ?  ?  l  6  ?  ?    }   ?  ?  ?  }  r  f  [  O  I  E  B  >  :  7  1  )            ?  K  /    ?  ?  ?  ?  ?  k  S  2  	  ?  ?  ?  a  2    ?  ?  ?  ?  ?  o  P  1    ?  ?  ?  ?  ?  ?  ?  ?  s  M  $  ?  ?  ~  S  ?  ?  ?  #  x  ?  ?  ?  ?  #    ?  ?  :  ?  p  	?                    	  ?  ?  ?  ?    _  =    ?  ?    D  :  0    
  ?  ?  ?  ?  ?  ?  j  T  >  &    ?  ?  ~  >  M  >  /      ?  ?  ?  ?  ?  l  M  .    ?  ?  ?  g    ?  ?  ?  ?  ?  ?  h  C    ?  ?  {  E    ?  ?  h    ?    ?  X  N  D  :  0  '  !      ?  ?  ?  ?  q  Q  3     ?   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  q  P  (  ?  ?  ?  /  ?  Q  ?  ?  q  ?  ?  ?  ?  ?  x  R  *  ?  ?  ?  k  6  ?  ?  ?  R    ?  ^  ?  }  o  b  U  K  @  6  )      ?  ?  ?  ?  ?  ?  }  R  '  
  
g  
\  
L  
8  
  	?  	v  	  ?    u  ?  J  h    c  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  m  \  J  ?  ?  ?  t  c  N  9      ?  ?  ?  ?  ]    ?  ?  [    ?  ?  ?  ?  ?  v  b  J  0    ?  ?  ?  ?  ?  U    ?  ?  J  Y  ?  ?  ?  s  b  Q  :       ?  ?  ?    [  3    ?  ?  ?  ^  
L  
?  
?  
?  
?  
?  
?  
  
Z  
  	?  	x  	  ?  !  ?  ?  p    i  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  b  	  ?  V     ?  Q  :  $    ?  ?  ?  ?  ?  ?  ?  v  Y  5    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  n  M  +  	  ?  ?  ?  ?  z  \  7    ?    ?  ?  ?  ?  ?  `  <    ?  ?  ?  [  &  ?  ?  _    ?  `  ?  ?  ?  ?  z  j  Z  J  8  #    ?  ?  ?  ?  ?  ?  ?  ?      ?  ?  ?  {  h  ^  S  G  :  .  "    
  ?  ?  ?  ?  ?  ?    ?  ?  ?  ?  f  7    ?  ?  ?  Z  &  ?  ?  m    ?  $  {  	?  	?  
  
N  
`  
n  
a  
F  
%  	?  	?  	y  	  ?  ?    
  ?  ?  3  ?  w  j  Z  E  /    ?  ?  ?  ?  m  @    ?  ?  ?  Q    ?       ?  ?  ?  o  H     ?  ?  ?  q  ,  ?  u  (  ?  ?  5  ?  ?  ?  ?  g  A    ?  ?  ?  w  T  )  ?  ?  ?  ^    ?  U    y  `  G  6  %    ?  ?  ?  ?  ?  ?    d  7    ?  ?  Y  
CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ??n??O?      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N?   max       P?5:      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ??/   max       =??P      ?  ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?         max       @FE?Q??     	?   ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??         max       @vq???R     	?  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @&         max       @O            |  3?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @??           ?  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??C?   max       >N?      ?  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A???   max       B1:?      ?  5?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A???   max       B1+h      ?  6?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =???   max       C?r?      ?  7?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >3e   max       C?w?      ?  8?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  9?   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      ?  :?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      ?  ;?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N?   max       P?ù      ?  <?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??4?J?   max       ??8?YJ??      ?  =?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ??j   max       =??`      ?  >?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?         max       @FE?Q??     	?  ??   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??         max       @vqG?z?     	?  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @)         max       @P?           |  R?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @?`          ?  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F3   max         F3      ?  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ????҈?   max       ??8?YJ??       T?      (                              )      ?   V            <      	   I      u       S   #   
         2                     %      T   .   	   O            	               ?      3   9   '               O?Ov??N)?N?N?!sOC	?Oݔ?N?V?O<N?gjN?8?Pl?O??P??lP?5:N?(XOH{(N?iO?WN???N?mPP]?N*??PR?O???P??O?N!??N? O
:?P,%tN?O?s'N?txN?I?Ns?mO??O?7?N??6Pn)oOC'?N???PKJ?O!??N	}FN??N??MN?O4?0N??fN?~hO???No?O1??O???OYXbN?T?N5??On?N??O9\???/??j??o??o?T???T???#?
??`B??`B??`B?o$?  <o<t?<49X<49X<49X<D??<T??<?o<?C?<?t?<???<??
<?1<ě?<ě?<???<??h<??h<?<?=o=o=o=o=o=+=+=C?=C?=?P=?w=#?
='??=,1=0 ?=H?9=P?`=Y?=Y?=]/=m?h=q??=q??=}??=?%=?%=??=??=??P!#.0<EIUWYURI@<0$#!?
#/2<@DGH></#
??
@BMOOQZQOLEB@@@@@@@@).55=;85-)WQVZZ[gt???????{tg[W'0UbnmrnhbUIB>0+*$)05?BFB>5)%KMLUanrzz??znaUUKKKK[WXY[chknqtuwtlhf[[[@BFNQ[gilig[NB@@@@@@????????????????????????????????????????noz??????? ??????zn?????5IPSHJ)????? ??%#      NIIN[gmt|???tg[YRNN<CO\cf\OIC<<<<<<<<<<?????????
?????????????????????????
"#$##
??????)U^^OB)???tpost?????yttttttttt16;N[t???????tg[N<61"#/3HYafhhaUH<2)$"~????????????????????
#/<CGGJ<1#???44:<=HKNH<4444444444	

##&''#
				????????????leem??????????????zldmn{????{ndddddddddd??????????? ????????????????????????????????????????EOR[`hrtkh[OEEEEEEEEzy?????????????????z??????&#"!????"%'),5BNPNLDB65)""""?????5;@@5)???????????????????????????!"/;=AB@=;/%"!!!!???????)2:BG)?????HOT]amtz?????zmeaWTH????????????????????`aemnz{????????znma`ZWX^ammmz~???zmaZZZZ?????????????????????????!## ????????????????????????????

??????????????????

????? 
""
)6BOOSSSOB6+) [RRSRRRST[hnt???{th[??????????????#*./-$##????????????????????#/<EHUafka^UD</#nlptz????????tnnnnnn#-4?B[gqtvutg[NB51)#?????????????????????????????????z??????EE!E+E9E;E7E.E*EED?D?D?D?D?D?D?EEEù????????????????ù÷öùùùùùùùù???????????????y?x?y???????????????????????ʾʾ;׾׾??׾ʾɾ???????????????????????????????????????????????????????????????????????????f?M?<?.?0?4?M?X?f?r????Z?f?g?f?b?f?m?f?c?Z?Q?M?B?I?M?T?Z?Z?Z?Z?????????????????????????????????????????A?M?Z?f?l?s?x?s?f?Z?O?M?A?<?4?1?4?A?A?A?n?o?{ŅŇŒŋŇ?{?n?m?f?e?j?n?n?n?n?n?n???(?4?;?^?f?b?J?4????????߽ҽ?????@?M?Y?f?r??????r?Y?M?@?4?'???	??'?@???????????????ùàÇ?l?Z?J?H?a?zï????????3?8?-?8?8??????C?6?-?hƅƑƧ???????????%?'????????????????????????*?4?1?.?*?&???????????????????`?g?i?b?`?T?R?S?T?V?`?`?`?`?`?`?`?`?`?`?)?5?N?g?l?r?l?g?b?[?N?5????????????)ìù??????????ùìàÓÐÓàééìììì?Z?f?s???????x?s?m?f?Z?R?R?Z?Z?Z?Z?Z?Z???????????????????m?G?;?.?(?$?%?,?G?`???????????????????????????????????????????6?O?[?q?v?t?h?[?J?6?)? ?? ?????????6?׾???"??	?????վʾ??????}?~?????????׿ѿݿ?????????ѿ????????~?~?????????ѿy?????????????????y?m?`?P?G?;?@?G?Q?V?y¦¨¦?????????????????s?g?`?g?i?s???????????????!?-?3?2?-?%?!??????????????????A?Z?g???????????????????g?A?(????*?A????	???????????????????????????????s?????????????????????s?m?h?i?c?^?e?s???????ùϹչܹ??ܹϹ̹ù????????????????A?N?P?R?V?S?N?A?;?5?.?0?5?5?A?A?A?A?A?A?H?N?U?_?a?m?a?U?H?F?@?@?H?H?H?H?H?H?H?H?<?H?R?Y?[?U?O?H?C?<?/?*?#?!?!?#?%?/?4?<?????????
??!?#??
????????????¼»¿???/?<?H?O?U?]?^?X?U?P?H?<?9?/?.?#?/?/?/?/?I?U?b?j?m?e?N?1?#?
???????????????#?0?IāčĚĥĦĳĳĸļĿĳĦĚčČā?}?y?zā?
??#?0?5?9?3?0?#??
??????????
?
?
?
?"?/?;?T?a?s?|?w?t?m?a?K?H?F?)????????"ĦĳĶĿ????ĿĻĦģĚĎčăćčđĚğĦ?B?E?O?V?P?O?B?;?7?A?B?B?B?B?B?B?B?B?B?B???(?4?6?A?A???8?4?(?'????????ŠŭŹ??????????????ŹŭŠŚŘŘŠŠŠŠ?????????????????????????????????????????5?B?K?N?[?g?t?y?z?t?g?[?N?H?B?9?2?2?.?5²¿??????????????¿²¦¥¤¦°²²²²?????????????????????????????????DoD{D?D?D?D?D?D?D?D?D?D?D?D{DmDfDgDdDbDoǔǡǭǷǮǭǡǔǈǑǔǔǔǔǔǔǔǔǔǔ???ʼռּݼ޼ݼּּʼ????????????????????'?4?@?M?Y?f?r????????????r?f?M?@?6?-?'?@?M?Y?\?^?]?Y?W?N?M?@?4?/?*?+?-?0?4?6?@???
??#?%?#? ??
??????????????????????????????????????????????????????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?EٺL?Y?e?n?r?}?r?m?e?Y?L?L?A?C?L?L?L?L?L?L?5?N?Z?g?i?l?i?g?d?Z?N?K?B???>?<?8?5?-?5 2 & M N M > < N B Q H ! _ 1 W W G | D f 4 L 7  a O 1 ; ] 6 g v e D Y u @ K W M 0 E \ 7 < 6  B ` ? 3 * L $ U m L b t & g    :  ?  T  1  ?  ?  ?  ?  0  ?  ?  C  ,  ?  ?  ?  ?  z    ?  ?  ?  Q  h  ?  ?  ?  ;  ?  /  ?  r  ?  %    ?  D  z  ?    ?      e  6      =  ?  ?  ?  p  g  x  )  +  ?  s  ?  ?  C?<?o;o?D???t?;ě?<D???o??o?o<#?
=#?
=t?>!??=ě?<?o<?/<e`B=??<?<???=?^5<?j>I?=Y?=??=q??=t?=t?=Y?=??
=?P=8Q?=<j=0 ?=0 ?=}??=?\)=<j==??
=<j==??=D??=q??=P?`=P?`=???=?O?=?7L>N?=?o=?G?=??=??=???=?C?=???=??=?E?B&_B IB?B?lB??B	??B&ӅB??BlpB?.B?5B!4B!?B?xB??B?B	>OB1:?B??B!??B$?|B1By?B?$B??B݄BK9B?nB^?B\?B>AB(?dBF$BxzBdoB8qB?'B?BUrB??B??A???BtA???B??BKA?zuB?
B5?B?B[NB??B?B?1B??Bx
B??B* Bk^B??Bd?B&+?B??B@B??B?8B	?+B&}?B5?B?FB?QB?vB!EGB!??B??BG?B?+B	cxB1+hB??B!?@B$??B??BB;B?B??B??BɡB2?B.B=?B ?B(?6B??B??B??BB?B?vB;BCtB@B?A???B??A??NB?/B@oA??zB??B?%B@WBCZB?.B~B??B??BDB?VB)?fBPIB?%B??@?
?C?`OA΢?ADAOcA???@㬾A>?bA?ӕA>??A?<A5Y?@?t_A?[gB3LA?)?A??Ah?lA???Ȃ8AB6Ai4?A?+A׭?APL?AwyoAl?A??A??@b`kA?&&@W?>AD.?=???A??A?^8A?n6A?A?A??A??yA?~?A???A??dA?V7A??A6??A?;Aqz?A??`A???A?V?C?ȴB?h@???@?? @?*?A?.A!?C?r????A?φ@?	?C?XA?y8A?AO.A?k?@??A>??A?TA>hA?v?A3"@?ÕA?M?B?`A?A???Ai?A?~A͎AB??Ah??A?o?A?=AR?<Av?gAlʯA???A???@cǈA??q@TJ?AE2>3eA???A??A?dA???Aā?A? oA?q?A??A?i?A??A??/A6??A???Ar4_A?&?A??A?wC??WB?_@??@?F?@ҪA?}?A!iC?w?????A?u$      (                              )      ?   V            =      	   J      v   !   S   $            3                     &      T   /   
   O            	               ?      4   9   (                                    %               '      ;   I            !         1      '   )   '   !            /                           3         3                                                                           #               #      #   =                     #         !                  /                           1         )                                                      NY@?O?jN)?N?N?!sOC	?O??WN?V?O<Nc?2N??O??Ou?vPۚP?ùN?(XO5kN?iOq??NB.N?mO???N*??O???O???O??OP??N!??N? O
:?P$pN?O?s'N?txN??NF??N?biOy?N??6PX?O?{N???P}?N???N	}FN??N??MN?O4?0N??N?~hO9p?No?OZO?O!??N?T?N5??N??N}??O??  ?  4  (    ?  ?    ?  ?  %  ?  a  ?  ?  	'  -  H  P  ?  ?  ?  $  ?    ?  
?  Y  y  ?  ?  ?    &  t  7  b  ^  ?  ?  	?  
?  ?  
?    D    q  %  ?  ?  ?  >  *  
L  	?  	?  6  ?  8  ?  Q??j?49X??o??o?T???T????`B??`B??`B?ě???o<t?<t?=?\)=+<49X<D??<D??=o<???<?C?=??<???=e`B<??h=<j=??<???<??h<??h=\)<?=o=o=+=+=#?
=?w=+=?w=#?
=?P=e`B=0 ?='??=,1=0 ?=H?9=P?`=aG?=Y?=??`=m?h=?7L=???=?\)=?%=?%=???=??w=???('01<CIKIC<0((((((((!#/29<>@@<3/#?
@BMOOQZQOLEB@@@@@@@@).55=;85-)WQVZZ[gt???????{tg[W#09ISbgjmhf`UICB0,'#)05?BFB>5)%KMLUanrzz??znaUUKKKKXYZ[ehimormhe[XXXXXXCGNS[ghkhg[NCCCCCCCC?????????????????????????????????????????????????????????????????)5EJC5)????? ??%#      JMN[gkt{???}tg[ZSONJ<CO\cf\OIC<<<<<<<<<<????????? 	
	??????????????????????????
"#$##
????)5IOPIB5)???tpost?????yttttttttt@<;;?EN[gt{??}tg[NH@$/<HTaddca_UH<6,*'#$?????????????????????????
#/7<<:/.#
??44:<=HKNH<4444444444	

##&''#
				????????????gm???????????????zngdmn{????{ndddddddddd??????????? ???????????????????????????????
?????????HOS[bhorih[OHHHHHHHH??????????????????????????????"%'),5BNPNLDB65)""""??????28>=4)????????????????????????!"/;=AB@=;/%"!!!!????????(*???????ZUVacmqz}?????zmiaZZ????????????????????`aemnz{????????znma`ZWX^ammmz~???zmaZZZZ?????????????????????????!## ????????????????????????????

??????????????????

????? 
""
#)6BINNLHB61)#ZXWWZ[ahntx}|zth][Z????????
???????#*./-$##????????????????????#(/<<HKHH</$#pnrt?????????tpppppp-/6ABN[gktttrg][NB5-????????????????????????????????????????EEEE!E*E,E-E*EEEED?D?D?D?D?D?D?Eù????????????????ù÷öùùùùùùùù???????????????y?x?y???????????????????????ʾʾ;׾׾??׾ʾɾ?????????????????????????????????????????????????????????????????????????????f?M?@?3?5?@?M?]?r????Z?f?g?f?b?f?m?f?c?Z?Q?M?B?I?M?T?Z?Z?Z?Z?????????????????????????????????????????M?Z?f?i?s?v?s?f?Z?M?A?<?A?C?M?M?M?M?M?M?n?{ńŇŐŉŇ?{?n?n?g?f?n?n?n?n?n?n?n?n??4?A?R?]?Z?V?A?4?(????????????????@?M?Y?f?r??????r?Y?M?@?4?'??	?
??'?@Óàù????????????ùìàÓÇ?z?p?m?o?zÓ??????&?(??)?*???Ƴ?u?T?UƁƑƟƳ???????????%?'???????????????????????*?-?/?-?*?%????????????????????`?g?i?b?`?T?R?S?T?V?`?`?`?`?`?`?`?`?`?`?)?5?B?N?[?]?_?[?S?N?5?)??????????)ùû??????ûùìàÓÓÓàìîôùùùù?Z?f?s???????x?s?m?f?Z?R?R?Z?Z?Z?Z?Z?Z?`?m?y???????????m?`?G?;?5?1?0?1?8?G?T?`??????????????????????????????????????????)?6?B?O?[?f?k?k?d?[?6?)?????	????????????ʾ??????????????????????׾??????Ŀѿݿ??????޿Ŀ??????????????????????`?m?y???????????????y?t?m?b?`?S?P?U?^?`¦¨¦?????????????????s?g?`?g?i?s???????????????!?-?3?2?-?%?!??????????????????Z?????????????????????s?A?&??? ?.?A?Z????	???????????????????????????????s?????????????????????s?m?h?i?c?^?e?s???????ùϹչܹ??ܹϹ̹ù????????????????A?N?O?Q?U?Q?N?A?>?5?0?1?5?8?A?A?A?A?A?A?H?L?U?^?a?j?a?U?H?H?B?C?H?H?H?H?H?H?H?H?/?<?H?P?P?H?G?=?<?4?/?%?&?+?/?/?/?/?/?/???????????
?????
??????????????¿???/?<?H?O?U?]?^?X?U?P?H?<?9?/?.?#?/?/?/?/?0?I?U?b?h?k?b?J?0?#?
???????????????#?0čĚĦİĳĵĸĳĲĦĚēčāĀ?{?}āćč?
??#?0?5?9?3?0?#??
??????????
?
?
?
?/?;?H?T?a?m?p?t?q?k?a?T?0????
??"?/ĚĦĳľĿ??ĿĸĳĦĜĚėčćĊčĔĚĚ?B?E?O?V?P?O?B?;?7?A?B?B?B?B?B?B?B?B?B?B???(?4?6?A?A???8?4?(?'????????ŠŭŹ??????????????ŹŭŠŚŘŘŠŠŠŠ?????????????????????????????????????????5?B?K?N?[?g?t?y?z?t?g?[?N?H?B?9?2?2?.?5²¿??????????????¿²¨¦¦¦±²²²²?????????????????????????????????D{D?D?D?D?D?D?D?D?D?D?D?D?DD{DrDsDvD{D{ǔǡǭǷǮǭǡǔǈǑǔǔǔǔǔǔǔǔǔǔ???ʼμּڼۼڼּʼ??????????????????????M?Y?f?r?????????????r?f?Y?M?@?8?@?F?M?@?M?Y?Y?[?[?Y?V?Q?M?@?5?4?.?/?0?3?4?:?@???
??#?%?#? ??
??????????????????????????????????????????????????????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E??L?Y?e?i?r?s?r?g?e?Y?Q?L?D?F?L?L?L?L?L?L?A?N?Z?g?g?i?k?h?g?\?Z?U?N?E?A?A?@?>?<?A 8 # M N M > ; N B N E # _ " G W B | > ? 4 8 7   R A 2 ; ] 6 f v e D V u I G W H ) E R 2 < 6  B ` 9 3  L ! F j L b G  Z    o  4  T  1  ?  ?  ?  ?  0  ?  ?  ?    M    ?  ?  z  ?  ?  ?  .  Q  q  ?  c  ?  ;  ?  /  '  r  ?  %  ?  ?  ?  ?  ?  ?  Y    ?    6      =  ?  ?  ?  ?  g  ,  O  ?  ?  s  ?  ?  t  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  F3  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  j  i  ?  ?      -  4  .      ?  ?  s  1  ?  ?  A  ?  @  ?  (  )            ?  ?  ?  ?    L  $  ?  ?  ?  ?  ?  ?    {  v  r  n  k  n  r  u  y  v  n  f  ^  V  M  D  :  1  (  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  c  I  .     ?   ?   ?   ?   ?   j  ?  ?  ?  ?  ?  ?  ?  ?  g  E     ?  ?  ?  q  4  ?  ?  
  ?  ?  ?      ?  ?  ?  ?  ?  ?  y  v  ?  {  _  =    ?  ?  ;  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  q  e  Y  L  @  )     ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  T  2          #  %  "              ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  g  <    ?  ?  ?  S  ?  ?  "  J  _  \  I  -  5  C  O  B    ?  ?  k    ?  	  h  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  n  R  -  ?  ?  S  ?  ?  B  ?  
c  a  ?  d  ?  (  t  ?  ?  ?  r  !  ?  ?  
?  	?  ?  P  ?  ?  ?  n  ?  	  	  	&  	   	  ?  ?  2  ?  /  ?  ?    8  Y  i  ?  -  #        ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  l  \  M  >  0  F  H  G  F  ?  2  "    ?  ?  ?  ?  ?  ?  k  E    ?  |  ?  P  L  H  D  ?  ;  7  3  .  *  #         ?   ?   ?   ?   ?   ?  (  ?  -  ?  ?  ?  ?  ?  ?  ?  2  ?  p    ?    b  ?  ?  (  ?  ?  ?  ?  ?  %  \  i  o  p  r  r  Y  @  %  	  ?  ?  ?  P  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  V  <    ?  ?  ?  ?  c  ;  ?  .  ?  ?    #      ?  ?  ?  ]    ?  t  ?  X  s  ?  k  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  k  ]  N  l  	  \  ?  ?      ?  ?  r    ?  ?      	?  ?  r  1  ?  ?  ?  ?  ?  ?  ?  ?  ?  k  >    ?  ?  =  ?  ?  +  ?  ^  N  	?  
0  
q  
?  
?  
?  
?  
?  
?  
[  
(  	?  	?  	7  ?  ,  ~  ?  ~  j  ?  ?       7  H  K  Q  P  5    ?  ?  v  7  ?  ?  X    B  y  |  ~  v  o  k  g  c  `  \  Z  Y  b  r  ?  ?  t  I    ?  ?  ?  ?  ?  u  g  W  F  5  $    ?  ?  ?  n  <  
  ?  ?  ?  ?  ?  ?  ?  ?  ?  h  K  +  	  ?  ?  ?  I  ?  ?  3  ?  _  ?  ?  ?  ?  ?  ?  ?  s  '  ?  *    ?  ?  ?  .  ?  ?    5  x       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  s  l  s  ?  ?  &  !      ?  ?  ?  ?  r  Q  ?  :  4  "    ?  ?  ?  ?  k  t  _  F  +  
  ?  ?  w  ;  ?  ?  ~  =  ?  ?  q  -  ?  ?  )  6  6  6  4  1  -  &        ?  ?  ?  ?  Y    ?  s     ?  _  a  b  d  g  k  u    ?  ?    v  j  ^  Q  B  ?  I  a  ~  ?  ?  +  J  Z  ]  Z  Q  E  3    ?  ?  \  ?    ?  9  |  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  W  +  ?  ?  G  ?  =  ?  ?  ?  ?  i  K  ,  
  ?    ?  ?  ?    M  	  ?  q     ?  u    	\  	?  	?  	p  	T  	1  	   ?  ?  s  d  a  )  ?  Z  ?  ?      ?  
?  
?  
?  
?  
?  
?  
?  
=  	?  	?  	?  ?      ?    L  ]  <  D  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    a  9    ?  ?  v  K  -    	?  
D  
n  
?  
?  
?  
?  
y  
G  
  	?  	?  	9  ?    G  <    ?  4  ?  ?        
  ?  ?  ?  ?  ?  ?  d  *  ?  g  ?  ?  |   ]  D  Q  ]  f  d  b  _  \  X  U  Q  M  H  C  =  6  /  )  $      ?  ?  ?  ?  ?  ?  |  `  D  (  
  ?  ?  w  S  #  ?  o    q  [  E  1    
  ?  ?  ?  ?  ?  ?  o  O  0    ?  ?  ?  ?  %        ?  ?  ?  ?  ?  ?  ?  ?  ?  t  a  N  ;  '      ?  ?  ?  w  h  r  u  ?  ?  ?  w  ^  9    ?  ?  I  ?  ?     ?  ?  ?  ?  ?  ?  m  Q  3    ?  ?  ?  o  7  ?  ?  u  3  ?  ?  ~  v  l  _  N  :  !  ?  ?  ?  ?  d  <    ?  ?  ?  ?  ?  ?  c  ?  j    7  >  4    ?  ?     m  ?  <  ?  B  w  ?  |  *    ?  ?  ?  ?  ?  ?  ?  ?  n  [  G  3      ?  ?  ?  ?  	?  
#  
G  
L  
E  
8  
"  
  	?  	?  	B  ?  ?    z  ?  ?  ?  e    	  	>  	q  	?  	?  	?  	?  	?  	?  	u  	?  	  ?  e  ?  y  ?  m  ?  ?  	?  	?  	?  	?  	?  	?  	?  	?  	O  	  ?  f    ?  7  ?  l    {  (  6  %    ?  ?  ?  ?  ?  ?  j  H  $  ?  ?  ?  S  ?  7  ?  ?  ?  ?  ?  t  e  P  <  '    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  2    ?  ?  z  ?    7  &  ?  ?  ^    ?  d    ?    Y    ?  ?  ?  ?  ?  ?  ?  ?  n  >    ?    #  ?  5  ?  -  ?    O  /  ?  G  0       ?  ?  ?  ?  T  "  ?  ?  v  6      

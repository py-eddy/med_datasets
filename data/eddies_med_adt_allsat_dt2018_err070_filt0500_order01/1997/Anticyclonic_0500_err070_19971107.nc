CDF       
      obs    -   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ??????+      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NQ   max       P???      ?  `   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?8Q?   max       =??F      ?     effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?z?G?   max       @D???
=q       ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       @33333@   max       @vf?Q??       &?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @-         max       @Q@           \  -?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @??           ?  .4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??o   max       >^5?      ?  .?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A???   max       B%?      ?  /?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?M   max       B%?_      ?  0P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @$\?   max       C?      ?  1   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @"??   max       C??`      ?  1?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  2l   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q      ?  3    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A      ?  3?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NQ   max       P?r      ?  4?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ????Fs?   max       ?܉?'RTa      ?  5<   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?t?   max       =??F      ?  5?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?z?G?   max       @Dٙ????       6?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       @p??
=?   max       @vf?Q??       =?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @)         max       @Q@           \  D?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @?.`          ?  E   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         DE   max         DE      ?  E?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???*0U3   max       ?܉?'RTa       Fx      3   (         :      	         k                  &                                    "      ,      ?   f         )   ,      "         6      ;OP?P;?PO???NQOF??P???N?`*N??]N?]O??JPH>N??O??GO??jN!?ONj??O?h?O@^?O?h?O6*)O?HN?Q?N?WKN?m?Ny?Na??O?OapO?? N?P=O?NH?P9??O?*O{=O#?O?x?Oݙ[N1??O`n6N?DO??O?BDN?aO?J??8Q?:?o;o;?`B;?`B<T??<e`B<e`B<?o<?t?<???<?9X<?9X<?j<???<???<?<?=C?=?w=#?
=#?
=#?
=,1=@?=H?9=L??=T??=Y?=aG?=u=u=}??=}??=?%=?+=?+=?C?=?hs=?hs=??w=?-=ě?=??`=??Fefgjnqt??????????the25;Wht????????th[G62!!!#,<H^ajib`UH<7/%!"(/;?;4/"9FN[lt????utga][NEB9????5BNZkH ???????????? 	

??????#$0<AEA<0#???? 

???????????>IThmz?????????tmTH>??????)5;@DFB5)??c\fgghtvx{?????tmhc????????,#
??????????????????????????????????????/06BOSQOCB?6////////????
#/219:5(
????79<AHIUanpxxtncaUH<7nr|???????????????tn\]acmz????????~zxme\????????????????????%)+5975)#????????????????????#0<FIU\UIB<60# 	


#*#"
????")+*)?????$,-,%#
 
$??????????????????ttqllt}???????mjipy?????????????tm??????????????????????????:Z^UJ5)?????????
???????????
#$##'(#
??zuvz}??????????????z?????????		???????BC@0#
???????
#/<B????????????????????????).6;=:5*)??	?
##,.#
						??????

?????jhdnz???????????xuqj????????

?????yusstz????????????zy?a?n?zÇÓàæâÑÇ?z?n?d?a?U?Q?K?K?U?a?M?Z?f????????????s?]?R?J?4?????8?M??????????????????????þý?????????a?a?g?m?q?t?m?i?a?\?\?_?a?a?a?a?a?a?a?a?y?????????????????{?T?R?Q?T?`?b?`?m?s?yƳ????????6?7?)???ƚ?u?O?6????*?C?hƳ???????ûĻȻƻû????????????????????????????????????????????~?|?z????????????????????	??????????????????????????????čĦĹ????????????????ĿĳĦĢĚĕđčč?
?#?<?I?S?Z?b?_?\?R?I?0?
????Ľ???????
?-?:?F?S?_?l?s?l?_?S?K?F?:?-?$?!??!?&?-?r????????????ɼ?????????x?r?f?^?[?d?r?ѿݿ????????????ݿѿĿ????????ǿ??a?f?n?x?r?n?a?_?V?`?a?a?a?a?a?a?a?a?a?a?ʼּ????ּܼʼǼ????????ʼʼʼʼʼʼʼ????/?C?T?]?_?[?a?T?H?/?"??	? ???????????m?z?????????????????????z?s?m?m?g?d?c?m?M?Z?f?s????????????t?f?Z?B?7?3?1?7?A?MŹ????????????????????ŹŭšŠŠŠŦŭŹ?????????????????????????????????|?z?????????????????????????????????????????????6?7?>?H?P?T?O?B?9?7?5?*?????"?)?4?6???????????????????????????޼ݼ?????(?4?A?M?Z?]?`?Z?M?A?4?/?(??????ĦĳĿ??ĿĿķĳĨĦěĚĦĦĦĦĦĦĦĦ?M?Z?f?o?j?m?i?f?^?Z?M?K?D?D?E?J?M?M?M?M??????????????}??????????????ʾ׾վʾ???!??#?? ??
??????????????????
???????????????????Ŀѿݿ??ݿۿѿпĿ¿????g?s?????????????????s?N?6?.?5?6?=?A?X?g???????? ???????????????????????)?6?B?H?L?E?6???????????????????D?D?D?D?D?D?D?D?D?D?D?D?D?DD{DsD{D?D?D??"?.?8?;?>?D?D?;?/?.?"?	????????	?? ?"?Z?g?s?z?????????????????s?g?Z?N?I?I?N?Z?????????????????????????????????????????g?N?B?)???
???)?5?B?[?a?t?w?s?w?u?g?????????????????????????????????????????????ɺѺԺҺʺɺĺ?????????????????????ǈǔǡǭǷǭǪǡǡǔǈǂǂǇǈǈǈǈǈǈ?*?6?;?C?P?\?i?h?[?O?C?B?6?/?*???? ?*?ɺֺ????ںֺкƺ???????????????????????EiEuE?E?E?E?E?E?E?E?E?E?E?EuEoEkEiEgEiEi?@?M?Y?f?r????~?r?m?M?@?'?????'?4?@ F [   ? G M  G F Y > X 9 6 @ + : 8 9 0 > M W \ } ; U @ B o ' Q @ < C j [ * M L J Z O A ;    ?  _    ?  ?  ?  ?  ?  ;  ?  ?  1  ?  >  T  ~  Y  ?  ?  ?  X  ?  8  <  ?  k  G  S  |  ,  &  r  X  L  Q  ?  ?  ?  l  ?  ?  {  ?  7  ???o=P?`='??<t?<?9X=?hs<???<?9X<??
=8Q?>   <?`B=L??=0 ?=+=t?=?C?=ix?=u=]/=m?h=8Q?=?+=e`B=P?`=u=?%=?o=? ?=?%=???=}??>^5?>'??=???=???=??=?`B=???=?
==?9X=?;d>?u>+>6E?B
?B??B?~A???B	?B??B#Y?B%?*B#??A?=B
xB1FB#?B-|BE?B$?B?5B?@B?QA??5B??BO~B?6B%?B$?B4?B??B$?hB?!B?B??B??B?B*B/UBLOB0?B5?B{oB{?BCB/?B?PBL?BmHB
>?B??B?"A?MB	??B??B#A?B%?_B#??B ?B<?B??B#=?B?$BA-B3#BE?B?TB??B ;?B??ByB?@B%?1B$??B?{B?
B$?"B??B?B?RB??B=B>B?^B<?B?LB@wB?RB??B'gB??B?)B??B?fA?čA>U?A?A?4,AlM?B??@?9P@??1@Q;2A??wA??%@?gT@?e?A~?^A??:A +?A???A?}fA@mA?oA??A??jA?I;A<|A:(QA?+A>?AL?A?9?AxMA??A??<A?]C??0A^?#A?ӝA??SA???A?[8@%??Bv?B ??@$\?C?@Յ?A?fA>|NA?a?A??`AmSB?&@?C@?Fk@O^A??TA???@?0?@? qA~??A?r?A ?A?o\A???AAg?A??A?=?A???A?mA??A7@IA??A?AL?A?W	Ax?!A?ȐA???A?ZC???A\??A?^?A?A?gZA?x?@#?YBA?B 9p@"??C??`@԰?      3   (         ;      	         k            	      '                                    "   	   ,      ?   g         )   -      #         6      <      3            Q            !   /      !            %      #                                    +      +               #               !                        A                                                                           +                     #                     N??O#??N???NQN]??P?rN???N??]N?]O_?=O?QN??OX?JO??N!?ONj??O???N??RO?&qO6*)Op-?N?Q?NA?N?)Ny?Na??O?N?	.O3??N?P=O?NH?O??CN??fO{=O#?O|wxO?:?N1??O`n6N?DO??O??N?aO?J?  3  ?  y  v  )  ?  ?  ?  ?  >  
?  ?  ?  ?  A  ?  ?  {  s  S  L    (  )  ?  x  ?  ?  ?  
  ?  ^  k  ]  ?  W  ?    ?  1  w  ?  	f  	?  
??t?<?`B<?9X;?`B<u<???<u<e`B<?o<???=?C?<?9X<?<?<???<???=\)=?P=#?
=?w=,1=#?
=H?9=8Q?=@?=H?9=L??=Y?=?o=aG?=u=u=??F=???=?%=?+=?\)=?\)=?hs=?hs=??w=?-=???=??`=??Fpjjlpt???????????tppLIKOT[`ht???}uih[OL../4<HISQHE<7/......"(/;?;4/"NNT[gjtwzttg[NNNNNNN????)5>GVQ= ????????????	?????????#$0<AEA<0#???? 

???????????TYahmpz????????zmaTT????)167750)??c\fgghtvx{?????tmhc????????

?????????????????????????????????????????????/06BOSQOCB?6////////???????
',55.#
???@@EHQUafnrsnmaUH@@@@xz?????????????????x\]acmz????????~zxme\????????????????????%)+5975)#????????????????????!#0<?IJI<<30# 	


#*#"
????")+*)?????
"#*,*#"
??????????????????ttqllt}???????mjipy?????????????tm?????????????????????????)1<@C<5)???????

???????????
#$##'(#
??zuvz}??????????????z???????????????????
#/<AB?/#
???????????????????????????).6;=:5*)??	?
##,.#
						??????

?????ywvsz?????????????}y????????

?????yusstz????????????zy?U?a?n?zÇÏÓÓÓÇ?z?z?n?a?\?U?Q?Q?U?U?A?M?Z?`?f?n?s?y?y?s?f?^?Z?M?A?4?3?4?=?A????????	?????????????????????????????a?a?g?m?q?t?m?i?a?\?\?_?a?a?a?a?a?a?a?a?m?m?y??????????y?m?l?h?m?m?m?m?m?m?m?mƳ??????????!?&?#???ƧƎ?u?\?9?3?=?\Ƴ?????????ûǻŻû????????????????????????????????????????????~?|?z????????????????????	??????????????????????????????ħĳ??????????????????ĿļĬĦģĞĜĥħ??#?0?<?@?E?C?<?0?#???????????????
??-?:?F?S?_?l?s?l?_?S?K?F?:?-?$?!??!?&?-?????????????????????????w?s?r?p?j?o??ѿݿ?????????????ݿѿ̿ȿʿѿѿѿ??a?f?n?x?r?n?a?_?V?`?a?a?a?a?a?a?a?a?a?a?ʼּ????ּܼʼǼ????????ʼʼʼʼʼʼʼ????	?"?/?=?K?T?Y?Z?T?H?/?"??
??????????z?????????????????????{?z?r?n?n?z?z?z?z?M?Z?b?s?z??????????s?f?Z?N?C?=?:?:?A?MŹ????????????????????ŹŭšŠŠŠŦŭŹ???????????????????????????????????}?????????????????????????????????????????????)?6?8?B?C?G?B?>?6?,?)?!? ?)?)?)?)?)?)?)????????
??
????????????????߼?????????(?4?A?M?Z?]?`?Z?M?A?4?/?(??????ĦĳĿ??ĿĿķĳĨĦěĚĦĦĦĦĦĦĦĦ?M?Z?f?o?j?m?i?f?^?Z?M?K?D?D?E?J?M?M?M?M?????????ʾѾҾʾ??????????????????????????
??????	?????????????????????????????????????????Ŀѿݿ??ݿۿѿпĿ¿????g?s?????????????????s?N?6?.?5?6?=?A?X?g???????? ????????????????????????)?1?9?9?6?0?)????????????????D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??"?.?8?;?>?D?D?;?/?.?"?	????????	?? ?"?Z?g?s?z?????????????????s?g?Z?N?I?I?N?Z?????????????????????????????????????????)?5?B?[?`?r?u?q?u?r?g?N?B?)??????)?????????????????????????????????????????????ɺѺԺҺʺɺĺ?????????????????????ǈǔǡǭǷǭǪǡǡǔǈǂǂǇǈǈǈǈǈǈ?*?6?;?C?P?\?i?h?[?O?C?B?6?/?*???? ?*???????ɺҺҺкκɺú???????????????????EiEuE?E?E?E?E?E?E?E?E?E?E?EuEoEkEiEgEiEi?@?M?Y?f?r????~?r?m?M?@?'?????'?4?@ ; 2  ? k L  G F J ) X : # @ + 3 , 7 0 4 M O F } ; U > * o ' Q . $ C j Q * M L J Z 5 A ;      `  ?  ?  ?  k  ?  ?  ;  ?  *  1  ?  F  T  ~  ?    .  ?  ?  ?  h  ?  ?  k  G    ?  ,  &  r  ?    Q  ?  7  ?  l  ?  ?  {    7  ?  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  DE  ?  ?  ?    '  -  3  .  "  
  ?  ?  ?  Z  '  ?  ?  ?  G  ?    >  I  F  5  "      ?  ?  ?  ?  ?  ?  t  <  ?  P  ?  ?  ?  ?  4  ~  ?  ?    C  d  v  x  j  @  	  ?  {     ?  =  ?  v  l  b  W  M  C  8  .  $            	        ?  ?  ?  ?  ?  ?  ?  ?        !  (  ,    ?  ?  ?  ?  ?  ?  :    8  i  ?  ?  ?  z  W  (  ?  ?  i    ?  .  ?  ?  X  ?   ?  ?  ?  ?  ?  ?  ?  o  \  G  0      ?  ?  ?  s  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  ]  A    ?  ?  ?  \  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  n  c  V  H  9  +    ?      /  :  >  8  +    ?  ?  ?  ?  Z  .  ?  ?  j  ?  k  	1  	?  
  
Q  
{  
?  
?  
?  
?  
?  
?  
?  
$  	?  	'  ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  p  c  U  E  5  $  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  b  2  ?  ?  c  q  t  z  ?  ?  ?  ?  ?  ?  ?  ?  o  R  -    ?  v    ?    A  C  F  H  K  M  N  N  O  P  P  O  N  L  K  J  J  J  I  I  ?  ?  ?  |  j  W  D  0    	  ?  ?  ?  ?  ?  ?  j  S  a  t  ?  ?  ?  ?  ?  ?  ?  d  =    ?  ?  x  <  ?  ?  ]  ?  ?  ,  m  j  g  r  y  {  w  o  ]  B    ?  ?  ?  2  ?  C  ?  ?   ?  Z  d  j  p  s  r  m  d  U  ?  %    ?  ?  a    ?  s    i  S  3  '         ?  ?  ?  ?  ?  q  I    ?  ?  ?  [  M  ?  	  G  K  I  B  :  /  "    ?  ?  ?  ?  j  -  ?  ?  C  ?  r        ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  ?  ?  	        .  A  ?  ?  ?  ?  ?  q  I  "  ?  ?  ?  ?    !  '  )         ?  ?  ?  ?  S    ?  ?  e  7  $  6  ?  v  h  Z  L  =  *      ?  ?  ?  ?  ?  m  R  7        ?  x  m  ^  E  0  !  	  ?  ?  ?  ?  ?  |  [  6    ?  ?  U  ?  ?  ?  ?  ?  ?  r  j  ?  ?  v  ]  6    ?  ?  M    ?  _    ?  ?  ?  {  o  `  B    ?  ?  ?  e  @    ?  ?  ?  ?  t  Q  T  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  9  ?  w  ?  7  c  ?  
  ?  ?  ?  ?  ?  ?  y  l  `  M  0    ?  ?  ?  u  I  '    ?  ?  ?  ?  l  N  3       ?  ?  ?  ?  j  3  ?  ?  R  ?  g  ^  Y  T  O  J  E  @  ;  6  1  *  "          ?  ?  ?  ?     ?  ?  [  ?    X  k  V  #  ?  f  ?    1  &  ?  
R  ?  g  ?  W  ?    ;  T  P  X  [  4  ?  u  ?  .  L  C  3    	?  ?  ?  ?  ?  ?  ?  b  B  "    ?  ?  ?  ?  |  Z  @  *    ?  ?  W  D  4  7  3    ?  ?  ?  ?  _  8    ?  ?  ?  ?  ?  ?  ?  ?  l  ?  d  9    ?  ?  ?  V    ?  ?  ;  ?  -  ?  ?  ?  ?      ?  ?  ?  w  I    ?  ?  @  ?  ?  ,  ?  ?     J  N    ?  ?  ?  ?  ?  ?  ?  ?  ?    h  N  3    ?  ?  ?  ?  ?  ?  1  )      ?  ?  ?  ?  b  )  ?  ?  ;  L  ?  ?      ?   ?  w  j  _  Y  R  J  B  9  1  )      ?  ?  ?  ?  l  F  ?  ?  ?  ?  ?  P  !  ?  ?  t  8  ?  ?  _    ?  ?  n    ?  \  "  ?  	
  	  	a  	P  	7  	  ?  ?  y  -  ?  {    ?  ?  D  ~  ?  O  	?  	?  	?  	?  	m  	>  	  ?  ?  h  ,  ?  ?  m  .  ?  ?  v  7  <  
?  
?  
?  
N  
  	?  	?  	`  	  ?  ?  >  ?  ?    y  ?  @  j  ]
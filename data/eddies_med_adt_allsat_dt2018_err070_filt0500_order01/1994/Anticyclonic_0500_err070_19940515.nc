CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ???t?j~?      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N??   max       P?-?      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ??9X   max       =Ƨ?      ?  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>Ǯz?H   max       @E??z?H     ?   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ?׮z?H    max       @vA?????     ?  (?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @'         max       @O@           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @?]           ?  1?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?49X   max       >A?7      ?  2?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?<   max       B,      ?  3?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A??)   max       B+??      ?  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?T?   max       C??      ?  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?RZ?   max       C???      ?  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          `      ?  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      ?  7?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      ?  8?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N??   max       P???      ?  9?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??ݗ?+j?   max       ??333334      ?  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?e`B   max       =??      ?  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?z?G?   max       @E??z?H     ?  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???G?z    max       @v@??
=p     ?  D?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @,         max       @O@           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @??           ?  M?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?M   max         ?M      ?  N?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???䎊q?   max       ??1???o     0  O?                           
   	   	      3   $   	                  ;                                 A         )                     `   =   
      R   
   J            
      A   ]O???N??{N"w?N?֊O???N?-sN,?FN*??NZ?O?O-?Ow??O??P??N(?yO>?tO???O??O&??P9KP?-?O?uO,?NO??O??N@?NݹgOP5GN???O\?O#B\P!??N?p?N?.?P??N?#N?QN.?N???O;?N?	nP/?P%@RNʗ?N?dP:??O(c?P#N??N??=N?B?N&?NE?yO??]O|h???9X?u?e`B?#?
???
???
?D??;o;o;?`B<o<o<t?<#?
<e`B<e`B<e`B<u<u<u<u<?o<?C?<?C?<??
<?1<?1<?9X<?9X<???<?`B<??h<??h<?<?<?<?<???<???=+=t?=#?
=#?
=,1=,1=D??=H?9=H?9=H?9=T??=Y?=?7L=??-=?E?=Ƨ?????????	????????efdgt????????tngeeeejnz??????ztnjjjjjjjj$"$)16BBJGB61)$$$$$$????????????????????????????????????????ZOS[dgomg[ZZZZZZZZZZ????????????????\dht????th\\\\\\\\\\QRRLNUajnoz?~zqnhaUQf`hlt????????????thfpq|??????????????ztp/HUanttnhaUH<8*#/AJSUKF;/("??????????????????????? 
#,/6;:3/#
????????????????)5KRW\\^[O5) %)-45BHHHHGDBA5)????????
?????????9E@9)?????????%5=DNZ[\_ggdd_][N5,%)557@@851) ????????????????????~{{???????????????~??????????????????????')))&??????????????????????????????????????????????????????????????????????????)5BR\ZNB)???????? ????????????????????????????
 /<NRN</#
????????????????????????w?????????????wwwwww????????????????????????????????????????RPPT[gt{????ttgf[RR???????????????????????????!0) ????21<?H[ht?????}th[B823,,15BCNP[^[YNB53333????????????????????~?????????????????~????????????????????? !,6BOZ[UURE6)?QPUbfgdbYUQQQQQQQQQQ~?????????????~~~~~~???? ????????{z~???????{{{{{{{{{{TNKITZadfaUTTTTTTTTT?????????????????????????????

???޻????????ûϻһ˻??????x?l?_?[?_?l?x????????????????????????üùîìù??????????????????????????????????????????????????ÇÓàåàÝÓÓÇ?z?x?t?zÂÇÇÇÇÇÇ?h?tāďęėčā?t?h?[?O?M?I?O?Q?Q?U?[?h??????????????????????????????????????????????????????????????????z???z?m?c?a?T?R?T?\?a?m?t?z?z?z?z?z?z?L?Y?\?d?`?Y?L?H?C?C?L?L?L?L?L?L?L?L?L?L?/?;?H?T?a?h?l?m?n?m?j?a?T?P?H?;?/?(?+?/?f?s?}?????????z?s?f?Z?M?K?M?Q?T?\?d?f???Ľݽ??????????ݽĽ?????????????????????"????????????????ú?????????T?a?q?s?m?a?H?;???????????????????"?;?T¿??????¿²§®²»¿¿¿¿¿¿¿¿¿¿?G?T?`?m?x?y?u?m?h?`?T?G?;?8?6?8?;?>?G?G???(?4?E?^?s?????s?f?Z?A??????????	??%?/?4?1?0?"??????????????????????	?????????(?*?1?(????????޿ݿ׿ݿ??H?T?m???????????m?c?Z?H?;?/??
??,?0?H?/?;?Q?j?Z?;?"?	?????????????????????	?/????$?0?>?I?[?V?=?%???????????????????M?Z?f?i?l?n?s?s?s?f?Z?M?E?A?:?A?B?F?M?M?(?4?A?M?N?P?M?A?4?-?(?!?(?(?(?(?(?(?(?(???*?6?;?C?K?C?>?8?6?*?????????
???#?+?#??
????????
?
?
?
?
?
?
?
?\?h?u?ƁƂƁƁ?u?m?h?e?\?R?O?O?O?Y?\?\??(?5?A?N?Z?f?s?k?g?Z?N?A?5?(????????????????????????????????{??????????????????'?.?9?2?'???????ܹչڹ?????????????????????????ùìàÓÔØàâìù????#?<?U?n?{ŋŎō?{?b?I?,?$??????? ????????ʾ׾????????????׾ʾþ??????????????????????????????????????????????????Ѿ????;ξ˾???????f?M?6?5?A?Z?l?~???????'?*?3?=?3?3?+?'???????'?'?'?'?'?'?#?)?/?6?9?4?1?/?(?#?????#?#?#?#?#?#ŔşŠťŭŭŹŽŹŭŠŕŔőŔŔŔŔŔŔ??"?.?.?8?3?.?.?%?"???	?????	????`?m?y???????????y?y?m?d?`?T?T?R?T?]?`?`?????ʼ˼ʼƼƼʼͼӼʼ??????????????????????лܻ????????????лû????~?o??????????????'?@?]?f?d?N?C?4????ֻܻ׻ݻػܻ?¦«§¦?z?z?~?(?5?9?A?B?F?A?5?5?(?%???????$?(?(àù????????????ãÌ?~?w?a?X?\?X?e?e?zà?нĽĽ??????????????????????????Ľнͽк????ɺֺ??????????ֺ????~?j?e?n?~?????!???????????????????
???????
????????????
?
?
?
?
?
?????????????????????????????????????????g?s???????|?s?g?a?`?g?g?g?g?g?g?g?g?g?g?b?n?{ŇŋŇŁ?{?n?g?b?b?b?b?b?b?b?b?b?b?????ʼּ??????????ּʼ?????????????????DoD{D?D?D?D?D?D?D?D?D?D?D?D?D?DxDpDqDmDo > / n M + . R s ] ? R ^  n E ) ^ 6 X 4 > a 9 B " U  Q B 2 \ F w N _ C } n 4  n B 9 C S W % T 5 | U 4 C D ,    4  ?  ?  ?    ?  -  ?  ?  C  m  [  ?  8  S  ?  ?  ?    ?  (  ?  /  q  O  T  ?  ?  ?  ?  ?  ?  S  ?  ?  ?  *  j  !  8       ?  ?    ?  m  ?    ?  ?  ?  i  ?  :?o?D???49X:?o<??
;??
:?o;??
<49X<?o<?C?<?`B=q??=<j<?9X=?P=,1=<j<?=,1=??<??h<??h<?1<???<?j=o=#?
<???=]/=ix?=?v?=?P=t?=?hs=\)='??=?P=?P=L??='??>
=q=??=P?`=T??>?=m?h=??m=]/=m?h=y?#=???=???>?>A?7B#	uB
9BdVB??B?	B?4B??B?RBB_iB?KBRgB??A?<B?xBx4B#(AB5BO B*B??B?B?ZB? B?B?B??B??B.?B??B!?0B??B?SBSBQ=B?mB?lB?tB!?B	H?B".?B?LB??B?UB+DuB?B,BU@B'?QBo?BB
??A???BF?B/,B#5<B
??BC'B?B??B̯B??B??Bc?B?B?IBCBχA??)BɒBBIB#R?B;B@+B?{B;BB??BK?B
²B?9B?PB?BE?B?B!?OB'?B+?B7mB??B1~B??B?B ?.B	MeB!??B<?B??B?tB+;?B?B+??B??B'?`B?{B0?B
G*A?R?BC?B?G@???A??A??A???A?P??T?A???A?(@??D?A??2A@??A'c?A?~?A??A? nAg}VA9R2A??6A?jCA?m?A?%?B	,?A??A9??A?
?A??BO?A?m?A???_ ?A??A???AQ?JA?:?AF?/??A??A??5A]??Al??@??@?;?@?lCA??A?g(A??A#-?@??A??(A??EA??QA?7?A?~@?g?C??@??kA?q^A?u?AɎ?A???RնA??YA?d.?҂?A?}AB??A&?&A҃8A??A?}LAhkA6??A??-A??$A??pA?}B?A>??A:+A??dA??tB;~A??A?|??RZ?A?t?A?;?AS
?A?[?AF????'AuA??A]?GAm ?@??@??@???A???A?[A?B?A#??@??A??;A??A?~?A?u?A??[@?+C???                           
   
   
      3   %   	                   ;                                 A   	      *                     `   >   
      R   
   K         	   
      A   ]   !                                 !      -            #      +   =   !                              +         +                     -   /         /      +                                                               #                  +   9   !                                       )                     %   /                                    O)N? ?N"w?N7?O#?NC:cN,?FN*??NZ?O?O-?O??NĳnO?n_N(?yO Ow1O?MO??P9KP???O?uN?foNO??O??N@?NݹgOP5GN???OD%TN?B^O??EN?p?N?.?P ?RN?#N?QN.?N??_NMQ?N?	nO?B?P%@RNʗ?N?dO}r@O(c?O???N??N??=N?B?N&?NE?yO??]O2,  ?  ?  ?  ?  ?  T  ?  (  i  U  y  ?  C  n  ?  ?  ?  S  ?  W  ?    a    <    ?  ?  ?  o  F  <  {  e  ?  ?  ?  ?      ?  
?  ?  ?  /  
?  Z  "  ?    ?  ?  ?  Q  ??T???e`B?e`B??`B;o?D???D??;o;o;?`B<o<T??=\)<?o<e`B<?t?<?t?<ě?<?o<u<???<?o<?1<?C?<??
<?1<?1<?9X<?9X<?/=C?=e`B<??h<?=\)<?<?<???=o='??=t?=m?h=#?
=,1=,1=? ?=H?9=??w=H?9=T??=Y?=?7L=??-=?E?=????????????? ????????fgegt???????tpgffffjnz??????ztnjjjjjjjj)$)6>BFB?64)))))))))????????????????????????????????????????ZOS[dgomg[ZZZZZZZZZZ????????????????\dht????th\\\\\\\\\\QRRLNUajnoz?~zqnhaUQf`hlt????????????thf|uv|??????????????||,+-/5<CHOUYUSH</,,,,"/;CNQPHA;/% ????????????????????
#(/276/.#
???????????????)5@INTRNIB5) !&).5BEGGHFCB=5-)????????
?????????????6B>7??????%5=DNZ[\_ggdd_][N5,%)57;50)????????????????????~{{???????????????~??????????????????????')))&??????????????????????????????????????????????????????????????????????)5:BGLOLB3)
?????? ????????????????????????????
,<LOH</#
????????????????????????w?????????????wwwwww????????????????????????????????????????X[\gotzvtg\[XXXXXXXX?????????????????????????????????21<?H[ht?????}th[B823,,15BCNP[^[YNB53333????????????????????????????????????????????????????????????
/6BFIIGB96)QPUbfgdbYUQQQQQQQQQQ~?????????????~~~~~~???? ????????{z~???????{{{{{{{{{{TNKITZadfaUTTTTTTTTT??????????????????????????

?????仑???????????ûƻû????????????}????????????????????????????ýùððù??????????????????????????????????????????????????ÇÓàÚÓÍÇ?|?z?x?zÆÇÇÇÇÇÇÇÇ?t?xāĉĎĒčČā?t?h?[?U?V?W?[?[?h?n?t???????????????????????????????????????????????????????????????????????????z???z?m?c?a?T?R?T?\?a?m?t?z?z?z?z?z?z?L?Y?\?d?`?Y?L?H?C?C?L?L?L?L?L?L?L?L?L?L?/?;?H?T?a?h?l?m?n?m?j?a?T?P?H?;?/?(?+?/?f?s?}?????????z?s?f?Z?M?K?M?Q?T?\?d?f?????Ľнݽݽ????????ݽнĽ???????????????????
????????????????????????????T?a?l?o?n?i?a?H?;?/? ?????????????"?;?T¿??????¿²§®²»¿¿¿¿¿¿¿¿¿¿?G?T?`?m?t?v?q?m?c?`?T?G?;?;?9?;?;?F?G?G???(?4?A?Z?w?r?f?M?A?4???????????????	??"?'?)?)?"?????????????????????????????(?-?(??????????ݿٿݿ߿??H?T?m???????????m?c?Z?H?;?/??
??,?0?H?????	??/???N?e?V?;?"?	????????????????????$?0?>?I?[?V?=?%???????????????????Z?_?f?h?j?h?f?Z?R?M?F?H?M?T?Z?Z?Z?Z?Z?Z?(?4?A?M?N?P?M?A?4?-?(?!?(?(?(?(?(?(?(?(???*?6?;?C?K?C?>?8?6?*?????????
???#?+?#??
????????
?
?
?
?
?
?
?
?\?h?u?ƁƂƁƁ?u?m?h?e?\?R?O?O?O?Y?\?\??(?5?A?N?Z?f?s?k?g?Z?N?A?5?(????????????????????????????????{?????????????????'?,?7?0?'???????ܹڹܹ??????ìù??????????ÿùìàÙØÝàêìììì?I?U?b?t?{?}?{?u?n?b?U?I?9?0?*?"? ?#?0?I???????ʾ׾????????????׾ʾþ??????????????????????????????????????????????????Ѿ????ʾ˾Ǿ???????f?M?F?B?I?Z?q?????????'?*?3?=?3?3?+?'???????'?'?'?'?'?'?#?)?/?6?9?4?1?/?(?#?????#?#?#?#?#?#ŔşŠťŭŭŹŽŹŭŠŕŔőŔŔŔŔŔŔ?"?'?-?.?0?.?+?"??
?	???	???"?"?"?"?y???????????y?m?i?g?m?o?y?y?y?y?y?y?y?y?????ʼ˼ʼƼƼʼͼӼʼ??????????????????????ûлܻ??????????ܻлû????}??????????????'?@?]?f?d?N?C?4????ֻܻ׻ݻػܻ?¦«§¦?z?z?~?(?5?9?A?B?F?A?5?5?(?%???????$?(?(Óàìù????????????ìàÓÇ?z?u?|ÅÇÓ?нĽĽ??????????????????????????Ľнͽк??????????ɺֺݺպɺ????????~?y?t?z???????!???????????????????
???????
????????????
?
?
?
?
?
?????????????????????????????????????????g?s???????|?s?g?a?`?g?g?g?g?g?g?g?g?g?g?b?n?{ŇŋŇŁ?{?n?g?b?b?b?b?b?b?b?b?b?b?????ʼּ??????????ּʼ?????????????????D?D?D?D?D?D?D?D?D?D?D?D?DD{DwDyD{D?D?D? 4 0 n V ' & R s ] ? R 0  b E - \ 2 R 4 8 a ; B " U  Q B 4 6 3 w N ^ C } n ; , n > 9 C S B % 0 5 | U 4 C D !    %  ?  ?  G  i  P  -  ?  ?  C  m  R  ?  m  S  :  "    `  ?  ?  ?  ?  q  O  T  ?  ?  ?  ?  ?  Q  S  ?  ?  ?  *  j  ?  _    X  ?  ?      m      ?  ?  ?  i  ?  s  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  ?M  %  =  l  }  ?  ?  ?  ?  ?  ?  ?  t  [  ?  %  	  ?  ?  ?    z  ?  ?  ?  |  p  ]  H  0      2  ?  V  ?  ?  ?  9  r  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  S  ?  ?  @  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  F    ?  ?  F    ?  ?  +  :  G  O  U  Y  Z  W  P  E  6       ?  ?  ?  Y    ?  e  ?  ?  ?  }  q  e  X  L  ?  3  &        ?  ?  ?  _  )   ?  (  !          ?  ?  ?  ?  ?  ?  ?  v  m  d  [  R  I  @  i  \  O  C  6  *           ?  ?  ?  ?  ?  ?  ?  ?  e  &  U  L  B  H  P  U  Y  W  R  L  E  @  >  9  /  #    ?  ?  ?  y  w  u  l  a  S  C  0      ?  ?  ?  ?  ?  ?  ?  s  "   ?  6  h  x  ?  ?  ?  ?  {  g  Q  5    ?  ?  ?  ?  _  ?  a   ?  ?  ?    G  y  ?  ?  ?     ;  C  <    ?  ?    ?    P  ?  ?    9  m  f  L  *  ?  ?  ?  E    (  ?  ?  c    ?  a  ?  ?  ?  ?  ?  ?  ?  ?  |  m  ]  M  >  3  F  Z  L  3    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  d  :    ?  ?  u  5  ?  T    ,  9  =  7  .  !    ?  ?  ?  u  4  ?  o  ?  ?  U  ?  [    '  8  F  O  R  S  M  >  "  ?  ?  z  ,  ?  |    ?  	  (  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  t  ]  7    ?  v    ?  9   ?  W  U  Q  M  H  =  +    ?  ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  l  U  ?  ?  x  *  ?  U  ?  -  .    ?  ?  ?  ?  ?  ?  ~  m  `  V  D  (  ?  ?  ?  ?  ?  J   ?  ?    (  =  L  W  ^  a  `  Z  S  I  /    ?  ?  R    ?  w    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  j  H  &     ?  <  5  -  !      ?  ?  ?  ?  ?  v  W  /    ?  ?  ?  ?  ?       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?        ?    u  i  ^  R  E  9  ,      ?  ?  ?  ?  A  ?  ?  L   ?  ?  ?  ?  ?  r  `  L  6       ?  ?  ?  @  ?  z    ?  /   ?  ?  ?  |  x  t  p  l  `  O  >  -       ?   ?   ?   ?   ?   ?   ?  ]  k  n  f  Y  H  8  *         ?  ?  ?  (  ?    {  ?  -    ?  ?  ?  E  .    ?  ?  ?  ?  ?  >  ?  ?    p  ?  ?  !  j  ?  ?  ?    !  /  6  <  )    ?  ?  F  ?    U  t    }  {  r  i  _  T  H  <  /  )  #    ?  ?  ?  ?  ?  ?  ?  ?  v  e  W  I  ;  *      ?  ?  ?  ?  ?  t  T  1    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  b  8    ?  ?  H  ?  ?  ?  ,  <  b  ?  ?  ?  s  c  R  A  /      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  p  [  F  4  #  ?  ?  ?  b  ,  ?  ?  ?  ?           ?  ?  ?  ?  ?  ?  ?  ?  u  d  S  C  2  !    	  
         ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  c  -   ?  ?  ?  ?  ?  ?  ?    	          ?  ?  ?  L  
  ?  -  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  s  c  T  P  R  T  V  V  V  U  T  
y  
?  
?  
?  
?  
?  
?  
?  
o  
&  	?  	~  	-  ?  J  ?  ?  ?    ,  ?  ?  _  "  ?  ?  ?  ?  ?  ?        ?  ?  A  ?     ?  ?  ?  ?  ?  ?  ~  e  J  0    ?  ?  ?  ?  ?  ?  ?  r  V  #  ?  /    ?  ?  ?  ?  ?  ?  n  X  A  )    ?  ?  ?  ?  L     ?  ?  0  	  	?  	?  
  
<  
`  
{  
?  
w  
K  	?  	?  ?  e  ?  ?  ]    Z  R  J  J  L  M  N  L  J  E  ?  7  -  !    ?  ?  ?  ?  ?  
      ?  ?  ?  ?         ?  ?  `  ?  h  ?  l  ?  ?  +  ?  ?  ?  ?  ?  ?  ?  ~  p  c  U  H  .    ?  ?  ?  ?  }  e    ?  ?  ?  ?  ?  ?  ?  ?  ?  v  [  ?     ?  ?  ?  ?  U  (  ?  ?  ?  ?  ?  o  [  F  2      ?  ?  ?  ?  U    ?  ?  K  ?  ?  ?      	              
  ?  ?  ?  ?  ?  h  K  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  j  Y  F  3     X  ?  ?  6  Q  <    
?  
?  
u  
-  	?  	?  	j  	  ?  g  ?  ]  ?  ?    _  ?  ?  Y  ?  ?  ?  u  ?    ?  k    ?    [  ?  ?  
y  ?  ?  ?
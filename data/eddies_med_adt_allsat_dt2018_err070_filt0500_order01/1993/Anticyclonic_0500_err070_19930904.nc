CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?pbM????   max       ??????+      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M?ٳ   max       Q?t      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?e`B   max       =??      ?  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>??Q??   max       @E*=p??
     ?   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??\(?    max       @vt??
=p     ?  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @0?        max       @P@           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?e        max       @??`          ?  2?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?o   max       >W
=      ?  3?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?Dr   max       B(??      ?  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?a?   max       B(??      ?  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =?{!   max       C?y?      ?  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?P}   max       C?xL      ?  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  7?   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q      ?  8?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %      ?  9?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M?ٳ   max       O?U?      ?  :?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ???a??e?   max       ?? ě??T      ?  ;?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?e`B   max       >C?      ?  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?\(?   max       @E*=p??
     ?  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ?ٙ????    max       @vtQ???     ?  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @0         max       @P@           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?e        max       @?}@          ?  O?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >P   max         >P      ?  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ??*?0??   max       ?? ě??T     ?  QX                  	               	         
   Y            :      b                  6      /      c         (   9                            ?                  H            Y         N??ObN?7?ND?O,elO ̈́N??O??N??8N'?O?k?N?6;N?X?N??_Q?tN??N4?N???PN<?O?{P??QN??	NC??O$?O?p?OmcO?f;O~?2Pj%O(?SPfI?OD?sO?tBO0??P6
MNa?/O???Nr?O???N??FN?a?OOqO?uO?ƛO3.?OF?N"}\N???OC??P?@O? N??No'?O??|O.ҜM?ٳN??e`B?T???t????
?o??o%   ;o;D??;?`B<o<o<#?
<49X<e`B<e`B<e`B<u<u<?o<?o<?C?<?C?<???<???<?j<?j<?j<ě?<ě?<???<???<???=?P=??=??=??=?w=?w='??=H?9=T??=Y?=]/=aG?=aG?=q??=y?#=?%=?C?=??
=?E?=?E?=?j=???=??=??????????????????????JJKN[gtu??ztkga[UNJ))6<BDB62)????????????????????_]\_amz|??????zmkda_TTUTU[hsty{vtslhb[TT" "#/<@DA<7/#""""""	5BN[bhhd][NB5dgt???????tqnljgdddd???????????????????????????	??????)5BINPQNB51)????????????????????#013500*#????? ?5V]???tB??????????????????????????????????????????88;<?HUaahaUH<888888???)6IUadbOB????|{?????????????????|??????? 	???????)),+)!')+5<BIDB5.)''''''''		"/99510/"	oqt??????????|~w}{voPJINUZanz|zxvtneaUPP|??????????????????|???????????????????????
#<ageaU</)	?
#&/0/*/04/#
?????)BV\TB"???  )36BOS[beb[OB>3.) jlt{?????????????utj???????????????????????????#$#+'????%)-45BNOPNB51)%%%%%%?????????
???????????????????????????0<IU[acab`[UIB0?????

???????????}??????????????????5589:BO[hsomh][OLB65khghmz??????????ztmk????????

????#/<=EHSSH>/#VVaajnz????????}zmaV{nnin{????{{{{{{{{{{jhnpz{????????znjjjj95/6;HTaehihaTQLH@;9?????)6FDB@A6)??)"???)6;<::61)????????????????????|?????????XVX^gt???????????g[X? 
#%),(#
?????????????????????????????????? ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????y?s?s?y?????????????????????????????????????????????????????????????????????u?y?????????M?Z?f?s???????~?s?f?Z?R?M?A?=?A?F?M?M?????????????????????????????????????????H?T?a?m?y?u?w???z?m?a?T?H?;?4?/?0?6?2?H?\?h?b?\?\?O?C?6?*?&?#?*?6?C?O?U?\?\?\?\??????????????????????????C?O?\?uƂƊƍƈƆƁ?u?h?\?O?0?*?&?(?6?CƎƒƗƚƖƎƇƁ?u?r?r?s?u?|ƁƃƎƎƎƎ?ѿݿ??????????????ݿտѿ̿οѿѿѿѼּ??????????????????ּռʼ??ʼͼּּּ??Z???????????	??	???????Z?(???????%?Z??"?/?;?E?B?;?0?/?$?"???	??	?????a?m?z????????z?m?j?a?W?a?a?a?a?a?a?a?a?????????????????????????????????????"?;?C?E?@?(?$??	???׾ʾ????????????	?"?????ſѿ????????ݿĿ????????????????????ľ?4?M???????????Z?A????????????????Ŀ???????????????????????????????????????m?v?y?~?y?t?m?`?]?[?`?h?m?m?m?m?m?m?m?m?H?T?a?a?m?n?v?v?m?a?T?H?;?9?0?3?;?D?H?H????????????????f?Z?4?*?0?4?A?M?f?s???E?E?E?E?E?E?FFF	E?E?E?E?E?E?E?E?E?E?E͹k?x?????Ϲܹ????????ܹù????????x?i?g?k??)?1?)????????????????????????????????????????????????v?s?z?????????H?T?a?g?n?p?m?c?a?T?H?;?/?"???"?,?<?H?B?N?g?t?y§§?g?B? ?????????(?B???ʼͼּؼۼڼּ˼??????????????????????<?H?L?U?X?`?`?U?H?<?/?&?#? ???#?/?1?<ùþ??????ùìàÔÓÇÆÀ?~ÇÓàìöùĦĳ?????
?#?+?@?<?0?
????ĿĩĐĊČĘĦ?I?M?U?b?d?k?c?b?U?R?K?K?I?C?I?I?I?I?I?I???7?@?M?P?S?Y?Q?@?3?-?'???????????r???????????r?r?o?r?r?r?r?r?r?r?r?r?r?Y?r???????????????r?f?Y?M?@?9?2?4?D?Y??#?/?;?5?/?(?#???????????????
???????
???????????????????????ûлܻ??????? ?????ܻ׻лλĻû?????ŠŭŹ????????źŹŮŭŪŠŖŔœőŔśŠD?D?D?D?D?D?D?D?D?D?D?D?D?D?D?DoD`DfDoD????? ?(?+?0?*?(??????????????????????(?+?6?;?:?5?(??	????????????????????????????????????????????????4?A?J?M?Y?Z?^?[?Z?M?A?9?4?0?0?3?4?4?4?4?tāĚĦĪııĬĦĚčā?x?v?t?h?e?h?l?t???!?:?F?_?u?}?????????x?l?S?:??????????@?L?]?e?v?~???????????????~?r?e?Y?Q?L?@?b?d?b?V?I?B???@?H?I?V?W?b?b?b?b?b?b?b?b?zÇÓàããàÓÇÀ?z?o?z?z?z?z?z?z?z?z??)?6?B?L?S?V?V?Q?E?6?"????????ǮǭǬǦǡǗǔǈ?{?o?f?b?h?o?u?{ǈǔǡǮE7ECEPEUE\EeE\EPENECE7E7E7E7E7E7E7E7E7E7EiEuE?E?E?E?E?E?E?E?E?E?EuEsEiEgEiEiEiEi : ! I P I = 4 C h a B X # # E 6 i U A 3 h Q 2 . G > r P h O E 2 ? Q V l Z 2 ' L W E 3 / / F 9 E $ M G J T H 9 ? Z    ?  D  ?  q    $  ?  ?  ?  I  k      ?  	;    u  ?  ?  8  ?  ?  X  \    5  ?    >  ?    ?  ?  ?  t  ?  ?  -  w  ?    ?  b  ?  ?  ?  A    ?  ?  B  ?  ?  ?  r  Q  ??o:?o???
:?o<t?;?`B;?`B<?j<o<#?
<?C?<D??<ě?<?1=??`<??
<?t?<?9X=??=t?=?l?<?j<?1=t?=t?=49X=??-=0 ?=?\)=??=??m=Y?=u=???=\=,1=?\)=49X=?+=T??=y?#=??=?O?>W
==??
=??w=???=?t?=?{>\)=?/=???=???>8Q?>J=?S?=??#B??B	B!?B LA??/B??BD}Be?B	??B??B??B7~B?7B%&fB=+BGxB??BO\B
#B?B"?lB-=BRA?DrBP?B@jB?cB_?B?MB?HB?Bd?B?RB"
B?Bs?B!??B#?1B&b?B+;Bz?B?qB 	JB !B+?BHNB(??B?CA??B?B?eB?B ]`B
e7B?!B?B??B>?B??B?/B?jA???B¯B@ B??B	??B??BC?B??B??B%=?BE?B??B?jB'JB<XB?8B"?xB?B@?A?a?B@?B<?B?B?\BËB<IB??BAnB?B">?B??BL?B!?B#?oB&FrBuBa?BM4B >?B։B?ABC$B(??B??A?{}B?B=?B?cB 5?B
@uB?$B?/B?&A?G?A??Ap??Z??A?r?A@ǝA??A??}B ?m@W]?B?uBjwA~??A3A???A??9A??"A?AAU?Av??A3??Ar"?Aj?IA???ACvC?y?=?{!A??A?3?A?c?A??B@???A?K?A???A?/A?M@?R8@?"@???A??A?+?@?_ A???C???A?_TA?<c@Xd?A;?Aތ@???@l?Bw?A??#A???B?C???C??nA?cA?\`Aod??U
?A??(AA`aA??yA??eB ?@^$?B?eBA7A~?}AUA?u?A?;A???A?	?ASJRAv?}A1??Ar??Aj??A?^?AC?C?xL>?P}A?#A?m?A?Z?A?D?@? YA?s?A?i?A??eA??@???@??@???A?mA?w?@?ޏA?y~C???A??A??@T??A;%{A?z?@??l???Bw?A??,A?s?B??C???C???                  	               
            Y            ;      b                  7      /      c         (   :      !                     ?                  I            Y                                                      Q            /   !   ?            #      %   !   1      9            /      %      #               !                  )                                                                                 !   #                     !                                 #                                 %                     N??N͊?N?7?ND?N??`O ̈́N??O?,?N??8N'?O?k?N?6;Nϥ?N\??O?N??N4?NX­O?^?O?[wO???N??	NC??N?ÿO?k?N?*?OaZO~?2OIA?O(?SO?GnO*?N?y?O??O???Na?/N???Nr?O???N??FN?a?O)?,O?uO?AO	?OF?N"}\N???OC??O?U?Ov?%N??No'?Oy3O.ҜM?ٳN?  b  ?  ?  ?    2  P  ?    J  g  r  ?  ?  P  ?  9  ?  6    ?  ?  ?  ?  k  d  ?  ?  ?  ?  	?  M    
  ?  s  ?  ?  \  d  h  ?  ?  ?  ?  }  0  ?  ?  ?  ?    ?  w  B  ?  ??e`B?#?
?t????
;o??o%   ;ě?;D??;?`B<o<o<49X<e`B=?C?<e`B<e`B<?o=49X<?C?=e`B<?C?<?C?<?j<?j<ě?=t?<?j=8Q?<ě?=?C?=+=?w=0 ?=m?h=??=L??=?w=?w='??=H?9=]/=Y?>C?=q??=aG?=q??=y?#=?%=??
=???=?E?=?E?=??=???=??=??????????????????????NNP[gnt{{trg[[ONNNNN))6<BDB62)????????????????????``ahmsz|??zymkca````TTUTU[hsty{vtslhb[TT" "#/<@DA<7/#"""""")5BN[_cc_[NB5dgt???????tqnljgdddd???????????????????????????	??????)5BINPQNB51)????????????????????#-020+#?????)0:=@@=5) ?????????????????????????????????????????98<HU^`UH<9999999999+6;DIKFB6)|??????????????????|?????????????????)),+)!')+5<BIDB5.)''''''''	"/451/+"z?????????????????RKINU[anzwusncaURRRR????????????????????????????????????????"!!#/<JUVYYWUOH</.%"
#&/0/*/04/#
)5BFGE@5)'))6BOQZ[`da\[OB60)'y~???????????????????????????????????????????	????%)-45BNOPNB51)%%%%%%??????????????????????????????????????0<IU[acab`[UIB0?????

???????????}??????????????????A69:;BLO[dholjh[YOBAkhghmz??????????ztmk????????

???????#/7<AHIHH<8/#VVaajnz????????}zmaV{nnin{????{{{{{{{{{{jhnpz{????????znjjjj95/6;HTaehihaTQLH@;9????)6<<>=6)????)06:;996.)%????????????????????|?????????Z\cgt???????????tg_Z? 
#%),(#
?????????????????????????????????? ?????????????????????????????????????????????????????????????????????????????????????????ſ????????????????????????y?s?s?y?????????????????????????????????????????????????????????????????????????????????M?Z?f?s???????~?s?f?Z?R?M?A?=?A?F?M?M?????????????????????????????????????????T?a?m?s?t?q?s?|?u?m?a?T?H?@?:?;?=?;?H?T?\?h?b?\?\?O?C?6?*?&?#?*?6?C?O?U?\?\?\?\??????????????????????????C?O?\?uƂƊƍƈƆƁ?u?h?\?O?0?*?&?(?6?CƎƒƗƚƖƎƇƁ?u?r?r?s?u?|ƁƃƎƎƎƎ?ѿݿ????????? ?????ݿ׿ѿοпѿѿѿѼּ??????????????ݼּ˼Ӽּּּּּּּ??Z?g?s???????????????s?Z?N?A?5?1?7?@?N?Z??"?/?;?E?B?;?0?/?$?"???	??	?????a?m?z????????z?m?j?a?W?a?a?a?a?a?a?a?a???????????????????????????????????????׾????	???	????????׾ʾƾ??????¾ʾ׿????ѿۿ??????ݿĿ??????????????????????ݽ????4?M?`?b?[?M?2???????ҽȽʽԽݿ???????????????????????????????????????m?v?y?~?y?t?m?`?]?[?`?h?m?m?m?m?m?m?m?m?;?H?T?a?k?m?r?r?m?a?T?M?H?=?;?;?;?;?;?;?????????????????s?f?Z?M?E?B?M?Z?f?s?E?E?E?E?E?E?FFE?E?E?E?E?E?E?E?E?E?E?E͹??????ùϹݹ????ݹϹù???????????????????)?1?)???????????????????????????????????????????????????????????????????H?T?a?g?n?p?m?c?a?T?H?;?/?"???"?,?<?H?N?[?g?t?z?w?g?[?N?B?,?"? ?&?5?B?N???ʼʼּڼؼּʼȼ??????????????????????/?<?H?L?U?V?U?T?H?D?<?0?/?/?&?#?#?#?-?/àìù????????ùìëàÓÇÄÂÇÐÓßàĦĳĿ???????
??
????????ĿĳĦĝĚĞĦ?I?M?U?b?d?k?c?b?U?R?K?K?I?C?I?I?I?I?I?I????'?*?4?9?=?4?.?'?"??????? ???r???????????r?r?o?r?r?r?r?r?r?r?r?r?r?Y?r???????????????r?f?Y?M?@?9?2?4?D?Y??#?/?;?5?/?(?#???????????????
???????
???????????????????????ûлܻ???????????????ܻٻһлǻû?ŠŭŹ????????źŹŮŭŪŠŖŔœőŔśŠD?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??????(?,?(?%???????????????????????(?+?6?;?:?5?(??	????????????????????????????????????????????????4?A?J?M?Y?Z?^?[?Z?M?A?9?4?0?0?3?4?4?4?4?tāĚĦĪııĬĦĚčā?x?v?t?h?e?h?l?t?!?:?S?_?n?x?????????l?F?:????????!?~?????????????????~?r?e?Y?S?L?D?L?_?r?~?b?d?b?V?I?B???@?H?I?V?W?b?b?b?b?b?b?b?b?zÇÓàããàÓÇÀ?z?o?z?z?z?z?z?z?z?z?)?6?B?G?O?R?R?O?L?B?6?)????????)ǮǭǬǦǡǗǔǈ?{?o?f?b?h?o?u?{ǈǔǡǮE7ECEPEUE\EeE\EPENECE7E7E7E7E7E7E7E7E7E7EiEuE?E?E?E?E?E?E?E?E?E?EuEsEiEgEiEiEiEi : + I P = = 4 L h a B X "   * 6 i Q & 1 K Q 2 3 9 4 O P $ O ) . @ O = l . 2 ' L W 2 3 ! " F 9 E $ C D J T > 9 ? Z    ?  ?  ?  q  ?  $  ?  4  ?  I  k    ?  e  ?    u  ?    ?    ?  X    	    ?    ?  ?  P  {    (  ?  ?    -  w  ?    o  b  *  0  ?  A    ?  =    ?  ?    r  Q  ?  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  b  a  `  _  V  J  ?  2  $    	  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    p  _  G  .      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  u  _  D  )     ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  t  c  P  =  (    ?  ?                  	  ?  ?  ?  ?  z  O    ?  ?  j  (  2  0  .  *  $        ?  ?  ?  ?  ?  ?  ?  e  I  *     ?  P  O  M  J  A  9  -      ?  ?  ?  ?  ?  t  Z  A  %     ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  f  M  -    ?  ?  P  ?            ?  ?  ?  ?  ?  ?  ?  ?  ?  }  e  M  0    ?  ?  J  =  1  %      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    g  c  ^  W  N  D  :  .  #         ?  ?  ?  ?  ?  ?  ?  x  r  p  o  m  l  h  [  N  B  5  %    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  c  +  ?  ?  ,  ?  i     ?  ^  m  |  ?  ?  ?  ?  ?  ?  z  k  W  >  #     ?  ?  ?  [  -  ?  H  ?  ?  ?  ?  ?  ?  ?     (  K  M  2  ?  ?  ?  M  M  `  ?  ?  ?  ?  ?  ?  ?  ?  s  d  U  F  7  &    ?  ?  ?  ?  ?  9  3  .  )  #        
    ?  ?  ?  ?  ?  ?  \  )  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  t  j  H    ?  h  ?  ?            $  2  5  !    ?  ?  m    Z  ?  ?            ?  ?  ?  ?  ?  ?  ?  x  X  4    ?  ?  n   ?  ?  s  ?  *  c  ?  ?  ?  ?  ?  ?  ?  _    ?  (  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  s  b  Q  ?  .    
  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  o  \  I  7  $     ?   ?   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  >    ?  ?  p  1  ?  ,  g  f  h  j  k  j  g  e  `  Y  N  E  4    ?  ?  ?  U   ?   v  N  c  Y  M  >  -      ?  ?  ?  p  8  ?  ?  F  ?  f  ?  t  }  ?  ?  ?  ?  ?  ?  o  ?    ?  ?  *  ?  ^  ?  @  ?  ?  ?  ?  ?  ?  ?  ?  z  h  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  o  e  ^  ?  ?  %  L  ?  ?  ?  ?  ?  ?  ?  ?  S    ?  a  ?  w  ?  `  ?  ?  ?  ?  ?  ?  ?  ?  r  ^  H  .    ?  ?  ?  d  5  !  6  ?  +  ?  ?  ?  	G  	?  	?  	?  	?  	?  	?  	h  	!  ?    b  ?  ?  C  1  A  M  H  9  "    ?  ?  ?  n  ?    ?  |  &  ?  K  ?    ?  U  ?  ?             ?  ?  ?  y  I    ?  ?  ?  ?    q  ?    
    ?  ?  ?  K    ?  ?  <  ?  m  ?  {    ?  Y  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  c    ?  D  ?  %  ?    s  k  c  [  S  I  8  '      ?  ?  ?  ?  ?  ?  c  5     ?  ?      n  ?  ?  ?  ?  ?  ?  ?  ?  e  G  !  ?  s    ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  n  ^  N  >  O  i  ?  ?  \  L  7  $    ?  ?  ?  ?  ?  ?  p  X  (  ?  ?  l  #  |   ?  d  \  S  K  B  8  0  )  !      ?  ?  ?  ?  ?  ?  c  6    h  c  [  O  9       ?  ?  ?  ?  ]  ;      ?  ?  ?  |  o  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  q  Y  >    ?  ?  ?  e  ?  ?  ?  ?  ?  ?  ?  l  J  #  ?  ?  ?  `  ,    
    %  ?  ?    ~  ?  K  ?  ?  7  r  ?  ?  ?  0  ?  ?  ?  E  ?  ?  	  k  ?  ?  ?  ?  ?  ?  ?  p  M  '  ?  ?  ?  Z    ?  u  ?  ?  }  w  m  c  T  @  $    ?  ?  ?  M    ?  ?  '  ?  U  ?  P  0    ?  ?  ?  x  O  $  ?  ?  ?  w  K  &  ?  ?  ?  ?  ?  9  ?  ?  ?  ?  ?  w  a  J  0    ?  ?  ?  s  T  8    ?  ?  ?  ?  ?  t  G  *         ?  ?  ?  ?  ?  ;  ?  ~    ?  D  ?  k  ?  ?  ?  ?  ?  ?  ?  h  ,  ?  ?  N    ?  a  ?  ?  ?   ?  ?  ?  ?  z  q  {  ?  |  h  L  $  ?  ?  K  ?  ?  c  ?  ?  1    ?  ?  ?  ~  l  Y  F  3       ?  ?  ?  ?  ?  ?  ?    v  ?  ?  ?  ?  m  G    ?  ?  o  .  ?  ?  h  "  ?  ?  D  ?  ?  ?  ?  ?  w  o  ^  <    ?  ?  %  ?  ?  #  <  
-  	  ?  b  ;  B    ?  ?  ?  \  /  ?  ?  ?  M    ?  m    ?  c    ?  ?  ?  ?  ?  p  Y  1  
  ?  ?  ?  e  ;    ?  ?  s  <     ?   ?  ?  ?  ?  \  0  ?  ?  ?  K    ?  ?  H  
  ?  ?  m  =  ?  ?
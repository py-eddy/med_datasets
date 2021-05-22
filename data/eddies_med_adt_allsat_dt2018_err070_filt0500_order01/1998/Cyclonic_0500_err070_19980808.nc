CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�XbM��     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�>�   max       P��p     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ȴ9   max       =�w     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F��G�{       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v
=p��       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       <�     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�    max       B4��     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}�   max       B4�c     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >0%K   max       C���     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?Uw   max       C��     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          c     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          Q     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�>�   max       P��p     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��S&�   max       ?�bM��     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ȴ9   max       =t�     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F��G�{       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v
=p��       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @��          4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         EW   max         EW     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�0��(�   max       ?��ߤ?�       cx   
         	                           8         c            @   	   `   !                        "               *                     #   1               !         (      *   H            9   J         	                      =         (            Nu��M�`�NT>�N܎�N�BN��N��O1a�OT�=N��Nu��N*�SP��pN��OK�iP���O+��NT��OB�P���Nc˽Pg�PC�O�}N3�*NJ�Nof�N[�O�}O��O���M�>�O�ZIN��O ��On��N��OO- �N]�%N�0�N;�N�p�O��P4�SOc��N�b�O��cP!�O���O��N��XO�*N�n&P _�P=�NmGNG�CM�+$P�SO̠�Nj�O5�N���N-  N�1�O��N!O�N(O�]�O�5ZN�E�OP�}O��fN�ZN ��O
3�N6z�=�w<�`B<�9X<#�
<#�
<t��ě���`B�t��t��t��#�
�49X�D����C���t����㼛�㼛�㼣�
���
���
��1��1��9X��9X��9X��j�ě��ě����ͼ��ͼ�/��/��`B��h���������o�o�+�C��C��C��\)�\)�\)�\)������w�#�
�#�
�''49X�8Q�P�`�T���Y��ixսq���u�y�#�}󶽁%��C���C����-���w��1��-��-��9X�ȴ9U[hmnlh[TPUUUUUUUUUU0<IUWUPI<90000000000<HQUanqnfaUKHB<<<<<<������������������������������������������

����������������������������&+.6=CO\g\VUQJEC*%'&	".;EHBB;</"
	BBO[^bdc_[YOMHDBBBBBS[htvttlh[USSSSSSSSS[[[gnptuutsge[[[[[[[���
#E^������{<#������������������������,/<HU]afdjcaUH<31,+,����	�������������}���������������|{}}���
���������������������������T^m����������umonVQT�������������������� 0Yfpw~�{nbI#
��� )18BNjt������t[B)!)�����#,/#�������u�����������uuuuuuuu����������������������������������������>BNY[d[[NNB@>>>>>>>>�th[WTPTX[bhltxz{|z����)3;6)!������������������wy������������������������;>HTaejebaZVTQNH=34;��������������������	
#/5783/-#

 		')5BN[gfd][PB51)$'��%������������������������������������������������#/<<HIKHB<4/#"?BNWTNFB86??????????���� 

������������
��������������������������|����������� ��������GOU[chnolmmhb[YPOHGG��������������������jm��������������mfdj0B[���������taNK=5/0dgnt���������|tpkgdd46<=ABO[hc][WOIDB>64��������������������������������������������6@HMJB������������**����������������������������TUYakha^UQPOTTTTTTTTQUbdeb`UTPQQQQQQQQQQ����������������������*-3434/&#
�����\anqyxna]\\\\\\\\\\\)1<@BGA5)!������������������������������������������������������������aanz����������zna\]a)26A862)��������������������f������������{ymf^^f��������������������"(),*)$
)5BEFB@<5)RVanz���������znaZRR//2<BHMH=<<72///////QUaenoona_VUQQQQQQQQHNP[gtttsnig\[[NMIGH:<GHQOHC<;88::::::::��������!�-�,�!������������������������ʼѼͼʼ������������������������������������������������������������
��������������������
����#�(�#��
�(�'�(�(�4�A�B�M�V�M�A�4�(�(�(�(�(�(�(�(�����������������ȼ���������������������²¨ª±²¿����¿µ²²²²²²²²²²��	����׾Ͼʾ׾�����	��"�$�3�.�"��	����������������������	���!����	�m�k�c�i�m�y�����������������y�p�m�m�m�m�������������Ŀѿӿ׿ѿĿ����������������ѿĿĿ��Ŀѿֿݿ���߿ݿѿѿѿѿѿѿ��	�������t�Z�5���&�Z��������������	�����������������ǾʾԾ׾۾۾۾׾ʾľ����������������������������������������������������������	�H�a�z���z�m�T�H�	����������������������������������������������z�m�m�a�\�a�m�z�����������������������������������������ʾҾ־;ʾľ�������ĮĦĬĿ���������
�<�|ņŃ�b�U�I�#����ĮŔōŊŔŜŠŭůŶŭŪŠŔŔŔŔŔŔŔŔ���y�`�b�y�����Ľн���������������н���ʾ��������Ⱦ׾����	������	���ܻӻͻλлû����ûܻ����8�5�'�����ìâàÝßàìòùûùóìììììììì�!����!�)�-�:�F�N�F�:�/�-�!�!�!�!�!�!�.�,�"����"�.�;�G�J�M�G�;�.�.�.�.�.�.�)��!�)�,�5�B�B�B�B�7�5�)�)�)�)�)�)�)�)�a�a�Y�W�T�H�;�/�"������"�/�;�H�T�a�ܹԹϹùǹϹйֹܹ�����������������ܻS�Z�U�a�l�x���������������������x�l�_�S�Y�Y�X�S�Y�^�f�r�y�r�f�[�Y�Y�Y�Y�Y�Y�Y�Y������������A�J�Z�g�v�s�g�b�N�5�(���������������$�&�(�$�"�������������H�H�<�:�2�5�<�H�U�a�k�n�w�v�n�m�a�U�H�H�=�;�3�.�-�*�0�=�I�b�e�o�s�q�o�h�b�V�I�=�s�n�r�s�}�������������������s�s�s�s�s�s�.�)�'�+�.�;�G�T�`�w�~�y�p�m�`�[�T�G�;�.�����������������������������������������)� �"�$�)�)�5�6�B�G�O�U�O�L�C�B�6�1�)�)ŔőŊŔŠŦŭűŭŠŔŔŔŔŔŔŔŔŔŔ��ýùöìéèìðù��������������������������ŷųŶ����������������������5������(�Z�s���������������s�g�N�5�ݿѿ����������������Ŀѿݿ���������ݹù����������ùϹܹ����������ܹ۹Ϲùÿ������������������̿ݿ�����ѿĿ���Ƴƨ�q�h�Q�Q�\�h�uƎƚƳ��������������Ƴ�z�h�b�g�n�o�z�������������������������z�����������������������������������޻ܻػлû��������ûлܻ����ܻڻܻ�ܺr�e�V�Z�k�r�������ɺ޺��ֺ��������~�r��������'�0�'�&���������������"����4�@�M�R�^�k�l�h�\�@�'��Y�^�y�������ʽ��!�%�%������㼱�r�Y�N�L�B�5�2�5�7�B�N�S�[�^�e�[�N�N�N�N�N�NE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��-�)�+�-�:�F�F�Q�F�:�-�-�-�-�-�-�-�-�-�-�~�l�j�r�~���������ɺֺ�������ɺ��~D�D�D�D�D�EE*E7EPE[EiEkEdEZEPECE*ED�D��5�(�0�5�A�N�X�V�N�A�5�5�5�5�5�5�5�5�5�5�[�V�T�[�h�t�{āčēėĚĦĦĚčā�t�h�[�лλû������ûлܻ�����������ܻл��O�L�H�O�[�h�o�o�h�[�O�O�O�O�O�O�O�O�O�OìèàÛØÚàìîù����������ùìììì�����������������ÿĿƿʿѿ׿׿ݿڿѿĿ�ìëèìðù��������ùùìììììììì�����x�l�i�l�l�x�}����������������������čĆĆčĚĳ������������
������ĳĚč������ĿĪĭĹĿ���������
��������������(�4�6�A�M�Q�Z�`�Z�M�A�5�4�(��ßÓÍÇÀ�{�w�r�zÇÖàìõ����÷ìáß�����������������������׾߾���ݾ׾ʾ�F$F#F$F)F1F8F=FHFJFJFJF=F1F*F$F$F$F$F$F$�����������ùϹӹҹϹù��������������������������������������
��#�/�/�$��
����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� : � � p V ; d n I Q + t � @ ! h   r + [ ? . @ N g _ o T : T ? e i ] ( . [ I Y ] e f ( D _ F l R : - 7 \ B w j T 4 H = Q ; A y C 5 V c O f  p > 9 t a V 6    o  �  �  :  V  *  O  �  �  �  �  h  	�  �  �  �  q  �  S    n  �  �  Z  {  u  ~  |  J  V  h  2  �  �  P  �  �  �  �    =  �  �  ;  ;    �  �  .  (  �  �  �  �  �  p  d    �    q  p    C  �  _  c  P  M  �  �  �  >  b  [  o  \<�<���<�o;o;ě�;��
�#�
�e`B��o�u�u�T����+���
�C���h��P�ě�����1��`B���Y��P�`�+�������ͼ����\)��P�m�h��`B�,1�\)�P�`��hs�\)�<j�C��L�ͽt���㽋C������Y��T���q����7L�'L�ͽ��w�P�`���`B�@��H�9�D�����`�   �ixս��
��+��o��������O߽�O߽����o�����/��󶽺^5��Q����S�Bn"B&��B��B�"B�B$YB�^B0��A� B�NB��B	W:B%8�B4��B9�B��BTWB�AB!��B ��B�xB%��B�<B#AAB
�B"/�B"��BA�BֱBlB ,B ?�A�<;B�B�SB�`B�!B�/B��B7�B��B��B�^B5�B�B7�B�3B z�B� B
B�B�B ţB"�]B�B-(�B�B��B'l4Bl�B?PB�zB�3BJ�B��BNB4�BN�B�0B
d|B�B��B��Bd�B��B,}B�*B��BD|B&�B��B@�B8oB$ <B�B0�JA�}�BìB��B	;B&�fB4�cB}sB��BITB��B!��B C�B�CB%AB	�B#@�B
��B">vB"�oB�B�!B�GB >XB ?9A�UB?�B��B�B`$B�BBƉB@OBȳB@*B?dB80B@LB>B[$B ��B��B
[B�B �)B"��B��B-@BÏB7wB'UwB?�B>�Bw�B�2B�!B��B��B�BF�B�YB
3�B;�B��B��BA�B�{BCB�&B� @dh�@�ʾA���A��<A:f@�a�A���AYLqA�>&AnrAx��A{��A���APaA�S�A�J�A��A��AMaA�L�A�z�A#fvAV�J@���A�M�@tv�AaXgA��A���>��6@���@ܠxA�+�BȎA�v4Bv(A���AfZZA��VAד�A�r�A���A���A���Ayp�>�1DAxPBt�A�A�b�@��@@�E?�j�@�Tx@�EhA�4;C�X@|V�@+�jC���A��qA�=�@�
�Aڜ_A̦�Ay	�A�@�@��FA��A�=�A9��A��PAN�vC���>0%KA��\C���@d{@��*A���A�j*A9�@���A���AX��A�:[Am4Ax�Ay[PA��IAN��A�n�A��iA�kwA�e{AMVA捈A��HA"o�AX��@�VÀ@sS�Aa�A��A��4?)�@��@�SA�B	?�A��+B��A��dAe��A���A��A�}�A΃oA�W�A���Ay�>���A{�BB�A���A�iB@�H�@+�)?�vD@��A��A���C�g@xE�@3��C��
A�#�A�l�@���A٨�A��FAx��A�l�@�`�A���A�A< mA�k�AN��C��>?UwA�T[C��i   
         	                           8   	      c            A   	   `   !                        "         	      +                     $   1                !         )      +   I      	      :   K         
                      =         )                                                   Q         9            A      1   )   +                     !                                    #   -             '   %         %      /   ;            +   #                           #                                                               Q         1            ;                                                                                    '                  %   5            +   !                           #                        NhVM�`�NT>�N܎�N�BN��N��O1a�OT�=N��NB�N*�SP��pN��O.
�PHb�N�,TNT��Nġ�P���Nc˽O���O�L�OZ��N3�*NJ�Nof�N[�O�}O��Os�fM�>�O0�<N��N�%�O.F�N��OO- �N]�%N�0�N;�Nk?ZO�IcO�Oc��N0�uO��IP!�O��(N��}N��}O�۾N�n&Oզ-P8=�NmGNG�CM�+$P��O�qXNj�N�s�N���N-  N��FN�+N!O�N(O�]�O�N�E�OP�}O��fN�ZN ��N�R�N6z�  �     �  �  �  �  �  �  Y  4    �  K    b  	Z  #  �  l  0  �  �    J  6  �  l  �  �    �  �  �  �  �  Q  6  �    �  7  �  )  �  �  �  �  �     @  �  �  �    �  
  T  ?  -  ?  �  �    �  J  �  �    {  	,  �  �  _  F  �  ;  9=t�<�`B<�9X<#�
<#�
<t��ě���`B�t��t��#�
�#�
�49X�D�������P��/���㼴9X��1���
�u�o��󶼴9X��9X��9X��j�ě��ě������ͼ���/�C���P���������o�+�C��}�C��'t��\)�,1�t���w�8Q��w�49X�'''49X�H�9�y�#�T���u�ixսq���y�#�}�}󶽁%��C����T���-���w��1��-��-��E��ȴ9Y[hhkhh[XSYYYYYYYYYY0<IUWUPI<90000000000<HQUanqnfaUKHB<<<<<<������������������������������������������

����������������������������&+.6=CO\g\VUQJEC*%'&	".;EHBB;</"
	BBO[^bdc_[YOMHDBBBBBV[hsrih[WTVVVVVVVVVV[[[gnptuutsge[[[[[[[���
#E^������{<#������������������������./<HU[aca`af_UH83.,.���������������������������������������
���������������������������Tam����������vnpoWQT��������������������
#0<ENRRPI;#

4;BN[dmt{����ti[N>44����
����������u�����������uuuuuuuu����������������������������������������>BNY[d[[NNB@>>>>>>>>�th[WTPTX[bhltxz{|z����)3;6)!�������������������������������������������8;>CHTU`aaa`\WTHA<78��������������������	
#/0450/'#
		$)-5BN[__[WNFB;5))#$��%������������������������������������������������#/<<HIKHB<4/#"?BNWTNFB86??????????����

������������
�������������������������������������� ��������LO[hhiha[ZOLLLLLLLLL��������������������jm��������������mfdjABN[gv�����t[NB>;67Ast����������tsmkssssBBOX[fb\[UOKFB=>BBBB�������������������������������������������)6BIE'�������������$))��������������������������TUYakha^UQPOTTTTTTTTQUbdeb`UTPQQQQQQQQQQ����������������������"'.0/0*#
�����\anqyxna]\\\\\\\\\\\)-57<;5)������������������������������������������������������������^acnz�������zna]^^^^)26A862)��������������������f������������{ymf^^f��������������������"(),*)$
)5BEFB@<5)RVanz���������znaZRR//2<BHMH=<<72///////QUaenoona_VUQQQQQQQQHNR[gsrmhga[NMIGHHHH:<GHQOHC<;88::::::::������!�"�&�!������������������������ʼѼͼʼ������������������������������������������������������������
��������������������
����#�(�#��
�(�'�(�(�4�A�B�M�V�M�A�4�(�(�(�(�(�(�(�(�����������������ȼ���������������������²¨ª±²¿����¿µ²²²²²²²²²²��	����׾Ͼʾ׾�����	��"�$�3�.�"��	����������������������	���!����	�m�k�c�i�m�y�����������������y�p�m�m�m�m���������ĿѿѿտѿĿ��������������������ѿĿĿ��Ŀѿֿݿ���߿ݿѿѿѿѿѿѿ��	�������t�Z�5���&�Z��������������	�����������������ǾʾԾ׾۾۾۾׾ʾľ������������������������������������������������������������������	�2�Q�R�N�C�/��������������������������������������������z�m�m�a�\�a�m�z�����������������������������������������Ⱦʾ̾ʾǾ���������ıĦĭĿ���������
�<�{ńł�b�U�I�#����ıŔōŊŔŜŠŭůŶŭŪŠŔŔŔŔŔŔŔŔ�������v�w�~�����������ƽн��׽нĽ�����׾Ǿ����ʾ׾ܾ����	������	����ܻڻֻܻܻܻ������'�&���������ìâàÝßàìòùûùóìììììììì�!����!�)�-�:�F�N�F�:�/�-�!�!�!�!�!�!�.�,�"����"�.�;�G�J�M�G�;�.�.�.�.�.�.�)��!�)�,�5�B�B�B�B�7�5�)�)�)�)�)�)�)�)�a�a�Y�W�T�H�;�/�"������"�/�;�H�T�a�ܹԹϹùǹϹйֹܹ�����������������ܻ����x�l�a�_�^�k�x�����������������������Y�Y�X�S�Y�^�f�r�y�r�f�[�Y�Y�Y�Y�Y�Y�Y�Y�(�"���	�����(�:�A�N�Z�c�Z�N�A�5�(��������������$�&�(�$�"�������������U�T�H�?�<�6�:�<�H�U�a�c�n�q�q�n�a�V�U�U�I�A�=�8�3�2�3�=�I�U�b�j�o�p�o�m�c�b�V�I�s�n�r�s�}�������������������s�s�s�s�s�s�.�)�'�+�.�;�G�T�`�w�~�y�p�m�`�[�T�G�;�.�����������������������������������������)� �"�$�)�)�5�6�B�G�O�U�O�L�C�B�6�1�)�)ŔőŊŔŠŦŭűŭŠŔŔŔŔŔŔŔŔŔŔ����úùìêêìïù��������������������������ŸŴŶ����������������������5�(�$�#�'�(�5�A�N�Z�e�g�g�g�\�Z�N�M�A�5�ݿѿ����������������Ŀѿݿ���������ݹù����¹ùϹعܹݹܹԹϹùùùùùùùÿѿĿ������������������Ŀ˿ݿ������Ƴƨ�q�h�Q�Q�\�h�uƎƚƳ��������������Ƴ�m�m�i�g�i�m�s�z���������������������z�m�����������������������������������ѻ����������ûлܻ�����ܻػлû��������~�t�j�j�r�~�������ɺֺ��ֺ����������~��������'�0�'�&�����������&�"�!�$�"����#�@�M�Y�\�i�j�e�Y�>�4�&�r�f�_�z�������ʼ���� �%�%�����㼱�r�N�L�B�5�2�5�7�B�N�S�[�^�e�[�N�N�N�N�N�NE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��-�)�+�-�:�F�F�Q�F�:�-�-�-�-�-�-�-�-�-�-�~�r�n�u�~���������ɺֺ�������ɺ��~D�D�D�D�D�D�EE*E7EPE\EeEaEWEPECE*EED��5�(�0�5�A�N�X�V�N�A�5�5�5�5�5�5�5�5�5�5�[�Z�X�[�f�h�tāĊČĄā�t�h�[�[�[�[�[�[�лλû������ûлܻ�����������ܻл��O�L�H�O�[�h�o�o�h�[�O�O�O�O�O�O�O�O�O�OìêàÜÙÜàêìù����ÿùìììììì�Ŀ��������������ĿǿѿֿտۿٿѿĿĿĿ�ìëèìðù��������ùùìììììììì�����x�l�i�l�l�x�}����������������������čĆĆčĚĳ������������
������ĳĚč����ĿĳĴ����������������
��������ؾ����(�4�6�A�M�Q�Z�`�Z�M�A�5�4�(��ßÓÍÇÀ�{�w�r�zÇÖàìõ����÷ìáß�����������������������׾߾���ݾ׾ʾ�F$F#F$F)F1F8F=FHFJFJFJF=F1F*F$F$F$F$F$F$�����������ùϹӹҹϹù����������������������������������
���"���
����������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� 0 � � p V ; d n I Q / t � @ " \ / r & [ ? " 2 * g _ o T : T = e T ] ' ) [ I Y ] e p  ! _ ' k R 0 . 0 _ B f f T 4 H 9 S ; 7 y C 6 K c O f  p > 9 t a B 6      �  �  :  V  *  O  �  �  �  S  h  	�  �  q  �  �  �  �    n  g  <  �  {  u  ~  |  J  V     2  �  �  �  t  �  �  �    =  �  �  Q  ;  M  |  �  M  �  �  9  �  I  �  p  d    �  �  q  �    C  �  1  c  P  M  �  �  �  >  b  [    \  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �               %  "          �  �  �  �  �  �  �  �  �    r  d  T  D  3  #      �  �  �  �  �  �  �  �  �  �  �  �  q  [  D  1  '      �  �  �  �  �  �    j  L  "  �  �  �  �  �  �  �  �  �  z  a  D  "     �  �  �  x  V  5    �  �  �  �  �  �  �  y  n  d  Y  L  @  3  '      �  �  �  �  �  �  }  j  W  D  0    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  g  Y  J  ;  !     �  Y  I  9  )      �  �  �  �  �  �  �  �  l  H  %      �   �  4  /  +  &  !        	     �  �  �  �  �  �  �  U  &   �                                
  �  �  �  �  �  �  �  �  �  �  �  {  k  W  B  .      �  �  �  �  �  �  �  K  5     �  p  J    �  j    �  �  q  E  !  �  �  e  �  -                      �  �  �  �  �  �  N     �   {  F  V  `  b  `  Z  N  7    �  �  �  R    �  �  p  B  +  �  E  �  	  	P  	Y  	L  	@  	=  	:  	*  	  �  q  "  �  X  �  �  �  �  �  �  �  �  �  �  �  �      	  �  �  �  a  <    �  �  W  �  �  �  �  �  �  �    {  s  k  c  b  g  l  q  {  �  �  �  G  R  \  d  i  l  h  b  U  E  2      �  �  �  o  5   �   �     "    �  �    U  +  �  �  h  X  N    �  `  �    s    �  �  �  �  �  �  �  }  o  a  S  D  5  #    �  �  �  �  f  &  �      $  Y  �  �  �  �  �  �  i  E    �     Z  ]  �  �  �                �  �  �  �  Q  G  1     �  r  �  ?  =  +  ,  3  >  I  C  1    �  �  �  {  =  �  �  �  �  {  6  P  c  e  ]  K  ,  	  �  �  �  P    �  �  e  '  �  �  f  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  =  	  �  l  f  `  Z  T  N  H  A  :  2  *  #       �   �   �   �   �   |  �  �  �  �  �  �  �  �  �  �  |  x  t  o  i  c  ]  W  P  J  �  �  �  �  e  E  1  !          !    �  �  �  �  �  b      �  �  �  �  �  x  U  /    �  �  �  �  �  �  �  �    ]  s  �  �  �  �  s  `  G  (     �  �  8  �  �  U  6  �  �  �  �  �  �    t  i  Z  J  :  *    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  Z  C  ,    �  �  �  �  �  |  g  �  �  �  �  �    t  i  T  >     �  �  |  *  �  �  1   �   �  �  �  �  �  �  �  �  �  �  �  y  D  
  �  ]  �  �  #  �  [  $  8  G  O  P  C  +    �  �  D  �  �  2  �  2  �  �  �  {  6  .  %        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  m  Z  @  $    �  �  �  v  L    �  �  O  �  �    }  |  z  y  w  v  n  c  X  N  C  8  0  /  .  -  +  *  )  �  �  �  v  b  M  8      �  �  {  1  �  �  5  �  y  �  u  7  &      �  �  �  �  �  �  �    l  Z  H  H  N  U  [  b  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  [  I  6  $    �  )  "    �  �  �  �  W  ,    �  �  �  K  �  �    x    b  }  �  �  �  �  �  �    �  �  �  �  �  Q    �  t    V  �  �  �  �  �  j  F  %      �  �  ]  "  �  �  �  g  X  J  *  9  C  M  b  �  �  �  �  �  �  �  �  �  �  i  :  
  �    �  �  �  �  �    u  j  Z  F  -      �  �  �  �  �  \  �  �    h  F      �  �  �  C  �  �  k  ;      �  �  2   �  �  �  �          �  �  �  p  7    �  �  k  '  �  �    =  >  ?  @  >  :  7  0  %          
    �  �        �  �  �  �  �  �  �  �  u  `  J  1    �  �  �  �  �  �  i  R  e  |  �  ~  q  \  >  !    �  �  �  �  Q    �  k  �  �  �    n  c  S  B  5  '    	  �  �  �  �  �  v  @    �  �  �      �  �  �  q  )  �  �  [  C    �  �  c    �       �  �  �  �  �  k  .  �  �  6  �  R  �  �  3  �    �  �  Z  
      �  �  �  �  �  �  �  �  d  =      �  �  �  �  �  T  R  O  E  5  '        �  �  �  �  �  �  �  h  C    �  ?  D  I  N  S  W  T  Q  N  K  I  H  G  F  E  H  M  Q  V  Z    +  &    �  �  �  �  f  (  �  �  T  �  �  -  �  +  _  w  �    5  >  0  �  �  d  
�  
�  
  	�  	�  	�  	f  �    2  	  �  �  �  �  �  �  �  �  �  �  w  j  ]  T  L  E  >  :  8  5  2  F  �  �  �  �  �  �  �  �  �  S    �  c  �  �  '  �  Q  �    �  �  �  �  |  m  _  R  A  -      �  �  �  �  �  �  �  �  �  �  �  �  �    j  U  A  .           %    �  �  �  @  G  I  G  E  C  B  =  4  +     	  �  �  �  �  k  @    �  �  �  �  �  �  �  �  i  M  *    �  �  �  �  ~  Q  �  �  4  �  �  �  �  �  �  l  P  5    �  �  �  s  6  �  �  (  �  F    
  �  �  �  �  �  �  �  l  P  3    �  �  �  �  x  X  7  {  v  a  8    �  �  �  s  �  i  m  _  G    �  w  �  �   �    �  	  	"  	+  	#  	  �  �  �  P    �  r  �  q  �  7  .  �  �  �  �  �  {  g  S  >  '       �  �  �  �  �  �  �  �  �  �  �  u  n  X  @  )    �  �  �  �  �  ^  -  �  �  �  0  �  _  ?  '  
  �  �  �  �  z  H    �  f    �  }  (  �  �  �  F  '    �  �  �  �  |  c  K  1    �  �  �  �  }  W  2    �  �  �  �  �  t  h  W  C  /      �  �  �  �  Z  0     �    2  -    �  �  �  �  l  ?    �  �  m  !  �  w    �    9    �  �  l  Z  C     �  �  �  y  R    �  �  �  K  �  �
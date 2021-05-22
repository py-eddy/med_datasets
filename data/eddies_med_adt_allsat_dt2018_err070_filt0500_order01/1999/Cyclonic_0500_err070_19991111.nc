CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�;dZ�     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�A   max       P�     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �   max       =�7L     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @F9�����     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��fffff    max       @v{\(�     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q            �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�p�         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �.{   max       =ix�     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0�     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��O   max       B0(     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >� �   max       C���     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C���     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          {     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�A   max       P�j�     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��YJ���   max       ?�"h	ԕ     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �1'   max       =�+     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @F+��Q�     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vz=p��
     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P�           �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��         (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G#   max         G#     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?�"h	ԕ        `�   
      
   
   z            H   R            d   B            #   0   
            &      
      P      M            	                                                   8         -   9         '                     7   '                  B   N�&7NR�%N��N�i�P�O��OX�N5N�Pi�PZT�N]��N�yN[VP���P{�NUhQO+dO*�VO���Oƅ�NȥNN�N	�N���P(��N'6,O��O0�UPD��Oi��P<�Oy��OTJN��NG��O3��OC�%OHc�NC�N1�vOh�M�AN��'O)�}O1&�OJ�hOxdO��NOzN�N��'O�3hO��O%˃O�TaO��O%.N��P �N��NG�\O|�,N��
O�n�O3�LO��FOG	�N@BO�"�OmۡN�׆N#0�O�Q�N�6=�7L=o<��
<#�
;�o;o��o��o��o��o���
�ě��49X�49X�T���u��o��C���t���t����㼣�
��9X��9X��j�ě����ͼ�/��`B��`B��h�����o�C��t��t���P��P��P�������'49X�H�9�H�9�H�9�H�9�P�`�Y��aG��aG��q���q����o��o��+��7L��C���C���C���C���hs��hs���㽛�㽟�w���
���� Ž�
=���������������������MOTZ[\hjhgb^[XONMMMM@BN[ac`[NCB=@@@@@@@@'/6<>HU`YUQH</$$''''�*-2B[����tg5����{���������������{|{{��������������������tt�������tqqttttttttJC����������CMOJ���)BKKP[B)������zz�������zyzzzzzzzz��������������������HOQ[hhjih[OMHHHHHHHH��� #0b{�����h<0������������������������������������������!#)/;HHLQRTX[TH;6/"!��������������������BHUnz�������zaUHC?@B	*6COT\]\UOC6
	Yadmz~����zma^Z[YYYY��������������������xz{������zvsxxxxxxxx�������������������������O[^XK6)�����]aemonmaUU]]]]]]]]]]��������������������#/<HUab`UTHF<2/#!�N[t�����tg[5 ���58;GIO[chjsxtg[OB655�������NWkd[5)����5BFIGED95)
���������������������

��������#)36?A6)&##########����������������������������������������#<>FKNNHE<4/+#
()/-,+))33686))|����������������{{|!#/0/'%#+/<>AFD<</.*((++++++��������������������55<BN[`edd[ZNIB>5325SV\amsz�����zmla]VS~������������������~�)25?8)������JN[gt����~tgg[QNGJJ��������������������hnnz�����zqnichhhhhh�����
���������!/Haz�����znaYSH;/#!
#07:20,&#
	!#(IUZ\^_\UI<50'$"!!jn}����������znhcbej����������������������

������������6IN[g�����zyt[B5,),6HHMT]adhkaTRIHHHHHHHoz������zxoooooooooo���������������������������������������X^^``gt���������tg[XZ[bgkt��������{g[YYZ������������#)/<?HKJGA<8/-#//3:<HOHHA<1////////���	 ������bn{|�������{naXXY]]b����������������������������������������!)5[g���������g5)!!

E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��r�l�e�Y�M�Y�c�e�p�r�~���������~�r�r�r�r����������������������������������������������������������������������������������ӿͿ��(�A�Y�k�v�������������������N��ܻԻû����ûлܻ�����"����������/�&�'�/�/�<�H�U�a�n�s�n�j�a�a�U�H�<�/�/�H�H�H�H�U�U�U�a�i�n�a�U�H�H�H�H�H�H�H�H�G�K�Y�Z�X�G�:���������������!�-�G���j�R�M�f��y������ϼ��ȼǼμ�߼ʼ����ֺԺкֺں�������ֺֺֺֺֺֺֺ�������������������������������������������������������������������������������������g�N�D�5�3�9�Z�s��������������m�`�T�B�<�@�G�m�����Ŀؿ������Ŀ��m�_�X�\�_�l�x�y���x�l�_�_�_�_�_�_�_�_�_�_��������������������������� � �����������g�]�g�m�s���������������������������s�g�H�=�2�/�'�'�+�;�T�T�a�z�������{�z�m�T�H�����ھվԾݾ���	�"�.�6�<�<�=�;�.�"���������������������	����������������6�2�*�����#�*�6�C�J�D�C�6�6�6�6�6�6����!�"�/�;�@�;�5�/�"���������ʼɼüżʼμּܼ���������ּʼʼʼ��	���������������	�"�;�M�N�R�O�H�;�/�"�	ÓÑÓÛàìíöìàÓÓÓÓÓÓÓÓÓÓ�g�f�`�Z�N�A�5�N�T�Z�s�������������s�l�g�6�)�(�)�(�)�4�6�B�N�O�Y�[�b�[�Y�P�O�B�6��������$�=�b�oǇǏǎ�}�o�Q�0�$����y�m�T�G�?�<�G�T�`�m�y�����������������T�H�A�4�"�����"�H�T�\�c�c�j�n�l�a�T������������������*�,�/�)��������߾�׾;¾ʾо������	���	� ���������������������'�)�-�'�������F�C�F�G�S�_�j�d�_�S�F�F�F�F�F�F�F�F�F�FŠŞŋŔŔŠūŭŹ����������������ŹŭŠ������������	��"�'�.�2�4�.�#�"��	�������������������������������������������лʻû»ûлܻ����ܻллллллл�ìæäìùý��������ùùìììììììì�a�Y�T�P�Q�T�]�a�m�z���������������z�m�aìèàßØàáìùúùóìììììììì�U�O�S�U�a�n�zÇÇÐÇ�z�n�a�U�U�U�U�U�Uù÷íÞÙÕÒØÕàäìù������ÿ��ÿù�
������������
��#�/�0�:�<�?�<�/�#��
�����������������������
����
��������#���#�'�0�7�<�I�U�b�d�b�^�Y�U�K�<�0�#�ƾ��¾ʾѾ޾������	�����	���׾ƾ���������	�������	������ŭŬŠŞŝŠŪŭųŹž��żŹŭŭŭŭŭŭ�Ŀ����������Ŀ˿ѿ׿ݿ�ݿѿĿĿĿĿĿĿݿÿ������Ŀѿ��������������������$�/�A�Z�g�j�j�e�[�A�5�-�(� �������������������������ĽнٽսнǽĽ������������(�4�A�O�Y�X�M�B�(������@�8�'�����3�@�Y�e�s�|���|�r�e�Y�L�@����ܹϹ������������ùϹܹ��������$���"�$�0�2�4�1�0�$�$�$�$�$�$�$�$�$�$��������������5�N�[�f�h�^�W�N�B�5�������Ƶ�����������������������������������������������������������������������e�\�Y�R�R�T�Y�f�r�������������������~�e�ɺ����������������ɺѺֺۺֺպ˺ɺɺɺ�����¿²�~�|²��������������/�,�#�������#�/�<�B�F�H�Q�L�H�<�/ùîàÛÖÚàìù����������������ùE�E�E�E�E�E�E�E�FFF$F/F1F3F1F'FFE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��3�7�9�7�9�G�S�`�t�������������y�d�Z�G�3�z�e�q�x�������ûлܻ�ܻٻлû��������z������������'�4�@�B�@�8�4�'�����#� ����#�0�7�1�0�#�#�#�#�#�#�#�#�#�#�h�_�^�l�o�i�k�tĄĚģĠĢĪıĤęč�t�hEEEE*E*E7ECEFECE:E7E*EEEEEEEE U Q ! W i \ D 8 I N I @ 2 H : 6 C � I * F � e E B % P H R : c 5 2 : < 7 + V Y T  d M p ' E T 7 M ? ` $ � 4 N D � H 7 A N  A P 8 ` B | b _ u X e i  �  }  �     $  �  a  X  5    �  �  y      W  {  J  �  �    �  b  �  4  @  d  �  �  �  �  �  M  �  b  u  �  �  >  j  �  2  �    u  �  i  3  M  �  �  �  �  `  I  _  ]  C  Q  �  �  �  �  @  �  �  �  y  r    �  :  �  f=ix�<���<49X%   �����t��T���o������ żD����1�u��S����w��`B��j���Y������h��j�����o�u�t��C��L�ͽ�"ѽ0 Ž�
=�49X���<j�,1�P�`�}�Y��,1�49X�}�#�
�8Q콇+��hs���w�y�#�����y�#�e`B�u��G���O߽��-�������ٽ����O߽�����T���㽾vɽ��T���ͽ������1�ě��Ƨ�9X��j�.{�$�B�aBNBT+BؤBA�Bk�BDKB��B.cB�B�0B{�B,�B&��B+�B �mA��B~zBf�B0�A���B��A��BհB�;A���BU�B�gB��B� B��B�SB�[B$5^B|�B�ZB$�B'kB�B%8BJ=B�%BpB��B�A��B �BZB	IiBwBE)Be�B�B%7B&^B�BEB�hB�A���B 
�B!�B#PB
B	�6B�B(�B�;BB(~dB*,�B6�B��B��B��B?!BJVB5AB��B@�B��B��B-C�B�EB�B�B�-B&��B*��B ��A��OBJ�B>cB0(A���B�A���B�B�KA�~�Be�B��B�QBϛBCBB�B�<B$>�B��B�"BD7B@,B%B6'B>�B��BE�B�B�6A�f�B ��B=;B	��B�
B=�B=�B?�B%?�B&ĵB��B8B�JB�GA�GB ?�B":�B#<�B
@B	��BQ^B@UB�iB^�B(<cB*:dBGAB`�B�)C�07?�NAt�A���A��q@��UAĴ�A�ڬA	�)@��@C��A��1A��OA���AsP3@�p~A�Y�A� A���A[jA�%kA��A���AA���A��'A�lA��BZ|Ak�A��7A�-HAVV.@�@���A�d�A\��A�Z�@�A�^&A�!�A�2�A�y�A�]fA��|A�$�A��AVAbAY[CA���Ayx^A~�A�'�A#�qA6fy?���>� �B	��A���Br�B��@s;@1w�A�~�A�AϚ�C���C�A�@�W�@�m�A�kxAݵXC��qC�(#?��DAt��A���A��O@�GdAáAŅLA�@� @C��A�|A��A���AmG@�0A���A�nA�k�AZ��A�}!B �:A��'A ��A��*A˱A��0Aغ�B��Al��A��A��,AV��@���@�e8A�u�A\�@A�}#@�<�A̎UA�5A���A���À�A�z�A��A���AU'MAY��A�~�AyBA}�A�j	A!W�A7�?ȫ�>���B	͗A�TyBRB:�@#�@3�/A�=A�A��C���C��A`m@���@ȿ�A��A�|C��Q         
      {            H   S            e   B            $   0               &      
      P      M            	                                                   8         .   :         (      	               7   (                  B                  I            6   9            ?   3            %   !               +            -      /                                                               !   #                  %               #                           #                  9            1   !            ;   3                              #            %      !                                                                  #                  %               #                              NHE�N)�N7ZLN�2^Pb��Nv�OOX�N5N�P@��O�?VN]��Nec�N[VP�j�Pk2NUhQO+dN��IO'��OX��NȥNN�N	�N���O�3�N'6,N޷�O0�UP	itOi��O��Oy��OTJN��NG��O3��OC�%OHc�NC�N1�vOA��M�AN��'N���N��O-��OxdO��N�B�N�N��'O��O��O%˃O]�O��O�N��P �Nl��NG�\OMBNC�O�{�O3�LO��FN�0N@BO�"�OR �N�׆N#0�On�N�6  1  :  %  �  0    ?  6  A  �  !  |  �  9  �     �  q  �  �  �  �  �  T  �  O  �  3  	T    	  #  �  �  �  &  �  g  �  �  P    r  �    =  �  6  \  �  \  |  Y  �  �  J  	  k  �  &  5  �  �  �  �  
�  	0    �    �  v    	=�+<��<�t�<t���h���
��o��o�T���\)���
�49X�49X�u��o�u��o��1�+�C����㼣�
��9X��9X��h�ě�������/�0 ż�`B�L�ͼ����o�C��t��t���P��P��P�'����H�9�H�9�T���H�9�ixսL�ͽP�`�Y��u�aG��q����o��o��+��+��7L��O߽�C���t���\)���P��hs���㽸Q콟�w���
���罧� ž1'���������������������NOV[hihe`\[ZPONNNNNN@BGN[]`\[NJB@@@@@@@@&/<@HU\XUOH</&&&&&&&�$,5mtug[B)���������������������������������������������tt�������tqqtttttttt������<EFD;�����������#��������zz�������zyzzzzzzzz��������������������HOQ[hhjih[OMHHHHHHHH�#0b{�����{e<0�������������������������������������������!#)/;HHLQRTX[TH;6/"!��������������������GHMUanz���|znnaULIHG*6CFNRRPIC6*Yadmz~����zma^Z[YYYY��������������������xz{������zvsxxxxxxxx�����������������������6JQPLE6)�����]aemonmaUU]]]]]]]]]]��������������������#/<HUab`UTHF<2/#!�)Ngtxzxqg[5'���58;GIO[chjsxtg[OB655���)5BIM5)�����5BFIGED95)
���������������������

��������#)36?A6)&##########����������������������������������������#<>FKNNHE<4/+#
()/-,+))33686))�����������������~�!#/0/'%#+/<>AFD<</.*((++++++��������������������458@BMN[[_^[VNB95444^ampz~�����zpma_XUX^~������������������~����������INQ[gt}����ztlg[RNII��������������������hnnz�����zqnichhhhhh��������������!/Haz�����znaYSH;/#!
#07:20,&#
	-0IQUW[\]ZUI<80)&%%-jn}����������znhcbej����������������������

������������6IN[g�����zyt[B5,),6NT\acga`TSJHNNNNNNNNoz������zxoooooooooo���������������������������� �����������b`bbgt����������tg[bZ[bgkt��������{g[YYZ������������"#/3;<@?<9/'# """"//3:<HOHHA<1////////���	 ������`bn{~�����{nbYYZ^^`����������������������������������������&)25;BNS[\d[UNB5*)%&

E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��r�o�e�[�d�e�j�r�~�������~�{�r�r�r�r�r�r���������������������������������������������������������������������������������������5�N�[���������������������s�N��ܻԻлλлܻ�����������ܻܻܻܻܻ��/�&�'�/�/�<�H�U�a�n�s�n�j�a�a�U�H�<�/�/�H�H�H�H�U�U�U�a�i�n�a�U�H�H�H�H�H�H�H�H��ּ������˼����!�.�:�G�P�S�S�L�:��Ｄ�����|�w�p�t�������������Ƽɼʼɼ������ֺԺкֺں�������ֺֺֺֺֺֺֺ���������������������	�	������������������������������������������������������g�P�F�8�9�B�Z�s�������������������m�`�T�D�?�>�C�G�T���Ŀտ����
������m�_�X�\�_�l�x�y���x�l�_�_�_�_�_�_�_�_�_�_��������������������������� � ���������������������������������������������������T�Q�H�@�>�@�@�H�T�a�m�o�z�z�y�u�m�a�]�T��	�����߾�����	��"�*�/�/�/�.�"��������������������	����������������6�2�*�����#�*�6�C�J�D�C�6�6�6�6�6�6����!�"�/�;�@�;�5�/�"���������ʼɼüżʼμּܼ���������ּʼʼʼ��������������	��"�/�;�F�H�N�H�;�/��	��ÓÑÓÛàìíöìàÓÓÓÓÓÓÓÓÓÓ�g�c�Z�P�V�Z�g�s�������������|�s�g�g�g�g�6�)�(�)�(�)�4�6�B�N�O�Y�[�b�[�Y�P�O�B�6�$� �#�#�%�1�<�I�b�o�{ǆǇ�w�o�b�I�=�0�$���y�m�T�G�?�<�G�T�`�m�y�����������������;�"����!�"�'�/�;�R�W�^�c�d�d�_�T�H�;������������������*�,�/�)��������߾�׾;¾ʾо������	���	� ���������������������'�)�-�'�������F�C�F�G�S�_�j�d�_�S�F�F�F�F�F�F�F�F�F�FŠŞŋŔŔŠūŭŹ����������������ŹŭŠ������������	��"�'�.�2�4�.�#�"��	�������������������������������������������лʻû»ûлܻ����ܻллллллл�ìæäìùý��������ùùìììììììì�a�_�V�T�R�S�T�`�m�z�������������~�z�m�aìèàßØàáìùúùóìììììììì�U�O�S�U�a�n�zÇÇÐÇ�z�n�a�U�U�U�U�U�UìåàÜÚÚßàìíùú��þùùìììì���
���������	�
��#�'�0�6�<�0�)�#���������������������
���
�	�������������#���#�'�0�7�<�I�U�b�d�b�^�Y�U�K�<�0�#�׾ϾʾǾɾʾ׾�����	���	�����׾׾�����������	�������	�����ŭŬŠŞŝŠŪŭųŹž��żŹŭŭŭŭŭŭ�Ŀ����������Ŀ˿ѿ׿ݿ�ݿѿĿĿĿĿĿĿ�ݿĿ������Ŀѿݿ�������������������$�/�A�Z�g�j�j�e�[�A�5�-�(� �������������������������ĽнٽսнǽĽ��������������(�4�A�K�V�T�M�=�(������@�8�'�����3�@�Y�e�s�|���|�r�e�Y�L�@�ܹҹϹ����������ùϹܹ������������$���"�$�0�2�4�1�0�$�$�$�$�$�$�$�$�$�$��������������5�N�[�f�h�^�W�N�B�5���ƶ���������������������������������������������������������������������������d�Y�W�U�W�Y�j�r�~�������������������~�d�ɺ��������������ɺκֺغֺҺɺɺɺɺɺ���¿²¦²�����������������/�,�#�������#�/�<�B�F�H�Q�L�H�<�/ùîàÛÖÚàìù����������������ùE�E�E�E�E�FFFF F$F,F$F FFE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��3�7�9�7�9�G�S�`�t�������������y�d�Z�G�3���~�i�u���������ûлڻڻ׻λû���������������������'�4�@�B�@�8�4�'�����#� ����#�0�7�1�0�#�#�#�#�#�#�#�#�#�#�t�n�h�f�h�j�r�tĀāčĘĚěĜęčċā�tEEEE*E*E7ECEFECE:E7E*EEEEEEEE H T / T l K D 8 F C I P 2 A B 6 C g ! % F � e E : % 3 H T : S 5 2 : < 7 + V Y T  d M \ ! A T 7 G ? `  � 4 : D o H 7 ; N  S O 8 ` 9 | b Y u X > i  i  S  U  �  �  �  a  X  �  �  �  �  y  �    W  {  �  k  �    �  b  �  C  @  �  �  �  �    �  M  �  b  u  �  �  >  j  �  2  �      �  i  J  -  �  �  �  �  `  �  _  �  C  Q    �  �  {  �  �  �  �  y  r  �  �  :  E  f  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#    '  1  *  !    �  �  �  �  �  �  m  B    �  �  �  O         ,  8  +      �  �  �  �  ]  /  �  �  �  j  <     �          $  %  $      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  Y  9    �  �  �  v  C  +    	z  
  
  
�     0  '    
�  
�  
@  	�  	�  	/  �  r  �  �  3  �  �  �              *  I  K  H  >  -    �  �  �  �  r  ?    �      �  �  �  �  �  �  }  \  8    �  �  r  M  9  6  /  (       	  �  �  �  �  �  �  �  �  �  q  `  d  q  ~    2  ?  =  .       �  �  z  ?    �  g    �  0  �  �  H  =  �    D  d  z  �  �  �  w  :    �  _  	  �  Y  �  �  �  !        �  �  �  �  �  r  L  $  �  �  �  `  -   �   �   �    3  G  W  f  s  z  |  }  �    z  s  e  H    �  u  #  �  �  �  �  �  �  �  �  �  t  _  H  1       �   �   �   �   �   o  .  7    �  �  �  |  g  S  8    �  x  
  �  �  [  �  �   �  �  �  �  �  �  �  �  �  ^  +  �  �  q  `  G    �  3  �  �       �  �  �  �  �  �  h  H  *  
  �  �  �  �  �  �  �    �  �  �  �  u  j  \  I  7  ,  "    
  �  �  �  �  t  G    P  G  5  #  U  l  m  T  6    �  �  �  �  �  h  <  �  �  �  �  �  7  V  q  �  �  �  �  �  �  �  �  �  b    �  �    �  4  T  a  m  w    �  �  |  g  I  '    �  �  <  �  [  �  b  �  �  �  �  �  �  �  f  H  &  �  �  �  �  \  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  z    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  T  D  4  &        �  �  �  �  �  �  �  �  �  T    �  �  �  �  �  �  �  �  �  �  h  b  [  R  G  D  B  2  �  _  �   �  O  .    �  �  �  b  5    �  �  q  >    �  �  q  <  �  �  n  �  �  �  �  �  �  �  �  q  ^  L  :  (      �  �  �  h  3  (  (  #    	  �  �  �  j  4  �  �  �  >  �  p  �  x  K  �  �  	+  	J  	S  	P  	K  	5  	  �  �  ]  �  n  �  �  �  �  ;  j    �  �  �  �  �  �  �  �  �  �  r  L  %  �  �  �  b   �   r  �  �  �  �  	
  	  	  	  �  �  �  �  O  �  ~  �  R  �  x    #      �  �  �  �  �  s  V  E  5  #    �  �  �  k    �  �  �  �  �  �  �  �  �  �  �  �  �  z  m  a  T  G  =  2  (  �  �  �  �  �  �  �  �  �  �  �  o  C    �  �  _    �  �  �  �  �  �  �  v  k  `  R  C  2    	  �  �  �  �  �  d  F  &  "           �  �  �  �  �  |  `  B    �  �  �  �  �  �  �  �  �  �  l  R  3    �  �  �  �  Y  )  �  �  U  �  S  g  Y  G  0    �  �  �  �  \  .  �  �  �  �  i  %  �  �  I  �  �  �  �  �  �  �  ~  d  5    �  �  �  e  @    �  �  l  �  �  �  �  �  `  E  1        �  �  �  �  �  �  �  �  �  H  O  P  M  F  6      �  �  �  f  8     �  Q  �  �  7  �                 �  �  �  �  �  �  �  �  �  �  �  �  �  r  e  X  J  ;  -      �  �  �  �  �  w  X  7    �  �  a  \  [  X  W  h  z  �  u  V  3    �  �  �  +  �  E  �  =  �  �  �  �  	      �  �  �  j  1  �  �  b    �  �  v  g  S  4  9  =  8  /    �  �  �  z  J    �  �  O  �  �  �  �  �  �  |  ^  @    �  �  �  t  c  D    �  �  �  h  O  R  [  h            2  3  "    �  �  �  �  b    �  -  �  �    U  Y  Y  Q  E  5     	  �  �  �  �  a  6    �  �  �  S    �  �  �  �  �  �  �  n  ]  O  A  3  (          �  �  �  \  U  O  G  >  5  ,  "    
  �  �  �  �  �  �  a  ?    �  q  v  z  i  S  <  "    �  �  t  &  �  ^  �  |    z  i  ~  Y  L  >  ,        �  �  �  �  �  �  �  ^  8    �  q  ,  �  �  �  �  �  �  �  �  �  n  V  ;    �  �  �  r  +  �  4  �  �  �  �  �  �  �  �  ]  .  �  �  j    �  ?  �  D  �  �  J    �  �  E  
�  
�  
L  	�  	�  	  �    �    ]  �  �  �  F  5  �  �  �  �  �  �  �  Y  !  �  �  /  �    g  �  �    0  k  h  e  b  _  \  Y  L  <  +      �  �  �  �  �  �  �  �  �  �  �  �  r  O  #  �  �  f    �  k  (  �  �  ]  �  �   �          �  �  �  ~  Q  !  �  �  �  V    �  �  c    �  5  #    �  �  �  �  �  w  Z  =       �  �  �  �  \  /    �  �  �  �  �  �  �  k  K    �  �  h    �  y     �  q    �  �  �  �  �  �  �  �  �  �  �  h  @    �  �  C  �  �  h  G  o    |  t  j  m  V  7    �  �  �  �  �  S  �  �  <  �  �  �  �  �  |  m  Z  F  /    �  �  �  �  x  O  (    �  �  
�  
�  
�  
�  
�  
  
_  
7  
  	�  	�  	I  �  X  �  �  �  }  C  �  �  	  	  	   	  	'  	+  	0  	   �  �  �  [    �  U  �  �    C    �  �  �  �  �  {  d  G  +    �  �  �  �  s  S  1     �  �  �  ~  �  �  v  M    �  �  �  W  +       �  �  c     )  �          �  �  �  �  �  �    b  E  %    �  �  �  O  �  �  �  �  �  �  �  �  s  e  Z  S  K  ?  ,      �  �  �  v  d  Q  ?  +      �  �  �  �  �  z  c  L  4      �  �  	�  	�  
   	�  
>  
E  
=  
9    
�  
�  
�  
3  	�  	K  �    !  �  %  	  �  �  �  �  S  %  �  �  �  s  ,  �  G  �  {    �  6  �
CDF       
      obs    S   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��1&�y     L  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�º   max       PcTR     L  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��E�   max       <���     L   D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>nz�G�   max       @F���R     �  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vy�����     �  .�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @R�           �  ;�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���         L  <(   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��h   max       <�9X     L  =t   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4ѹ     L  >�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4�=     L  @   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >%%   max       C���     L  AX   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >>�s   max       C���     L  B�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          8     L  C�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          7     L  E<   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          5     L  F�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�º   max       PcTR     L  G�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊q�   max       ?���8�YK     L  I    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��E�   max       <���     L  Jl   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>nz�G�   max       @F���R     �  K�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @vy�����     �  X�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @R�           �  e�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���         L  fP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�     L  g�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ߤ?�   max       ?ͿH˒;     �  h�               
                         	   $                                       	                     &                     	         8                   	                         .   
               "                               
                  
      /   Nf�N�gO`l'OP�&NѝBN��7O�5O`WLN&mO��NA�@O��pN���O&O�o�Np%�O�8�OُOT�jOb��OEx�P#p�O�'FO'[N_�`N]�N���O_��N���NaKgN�O��N{��PcTRN���N�?MN��N���N /oN:��O4�P8ŗO<
�O�pN�G�M�ºONNN���OB�O�N&�Oe��On��OfU�N���N�� O7^�Po#N�{�O�{�M�
�O&�O/�P!O+�,O�uZO�IOn�TO�1}O��N,ŖN{@�N��N�
O�:�O�GN7��N���N WtN��}O/��O�NCE�<���<e`B<49X;ě�;ě�;ě�;��
;D��;D��;o;o��o�D����`B��`B��`B��`B�o�o�#�
�49X�49X�e`B�u�u�u��C���t���t���t���t����
��9X��9X��9X��j�ě��ě��ě����ͼ�/��/��/��/��h�o�o�C��C��C��C��\)�\)��P��P��P��P���#�
�#�
�'''49X�8Q�D���P�`�P�`�P�`�P�`�P�`�T���aG��e`B�e`B�ixսu��%��%��%��O߽�O߽�E�������������������������������������������������������������������������������������������������������������������������)5B[m}zqg[N97.0-)����
#,//./*#
���nnpz����znnnnnnnnnnn�����������������������������������z�����������~zqmpsz����� � ����������������������������������
#,20&

�������6BOU[][[OMB866666666���������������������������� ������#/<HQRSSQLH</%#SUY_agmz~���zmga[VSS��� 	�������)Ohv~�ssoms[B6*6CO\gu~ue\C6*yz{��������������zy�������������������������������������

"'#
����/58BMOT[gostg[NG<50/���������

��������� 	


�����������[[httutkh`^[[[[[[[[[)5Ngpnga_lpe[NB;!#'/;5/#	#b{�����sU<0
	����������������������������������������6;HT^aca[TIHG=;56666���������������������������������������������������������STamouz{{zuma^TWTSTS���5BQQB94�����������		����������������������������������� ����������������������������������������������������GHUZafnqnja]UPNHGGGG'6:BGB63)
U[itz�����}vthb[XQSU�������������������������
#&)*(#
����	#/<HUY\]UH</#
	����������������������������������������DO[hpqnh[WOGDDDDDDDDy{������������}{xxy�):;+&*/<B5)����_adntz��zna`________���#0<BEEB;0#
���7BIOZ[\_[UOB77777777������

	�����,02<@IUWUTYZXURI<0-,
#<GUYWXUI<#
�������������������������������������
#+/24799/-#	:<Uanqv~~zrm]UPH><;:��������������������&13/$�����(0<CIKUUUIB<0/((((((��������������������NOU[hjpnihh[QOJGGMNN45BJNSQQPNNB55324444Wddgt���������ytg`TW���������������������������������)2/)!������BBJN[d`[NBBBBBBBBBBB������������������������������������������������������:<HUUVULHG<4::::::::�`�_�]�`�l�p�t�m�l�`�`�`�`�`�`�`�`�`�`�`�s�l�g�^�\�d�g�r�s�}�~�y�s�s�s�s�s�s�s�s�
��������
��#�/�3�<�?�H�U�U�<�/�#��
�������������������	��"�.�"���������׽��������������������������U�M�H�@�<�9�<�>�U�_�a�f�j�j�n�u�n�f�a�U������	��4�M�f�s�z�~��f�Z�A�4�(�!��Z�N�A�=�5�0�.�3�5�A�N�Z�g�s�}�|�v�m�g�Z�U�T�H�D�C�H�U�X�X�U�U�U�U�U�U�U�U�U�U�U��ƸƳƫƳ��������$�0�7�9�0�$��������̾��������������ʾԾԾξʾ���������������àÓ�z�t�n�l�l�zÇàì������������ùìà�M�L�A�?�9�A�M�Z�f�s�w�z�s�f�\�Z�M�M�M�MD�D�D�D�D�D�D�D�D�D�EEEEEED�D�D�D�àØÕÓÔÚàìù������������������ìà�a�U�Z�a�l�n�zÆ�|�z�t�n�a�a�a�a�a�a�a�a�M�E�J�I�C�H�M�Z�s����������������f�Z�M�z�t�m�j�t¤¦©­¦��������������������������� �����������t�h�[�O�E�O�Q�\�hčĚĦĮĭĦĢĚčā�t�����������������+�5�B�I�B�5�1������"��������	�"�;�G�`�y�����s�m�`�T�.�"�	�����������"�(�.�0�2�3�3�.�"��	�C�>�6�*���	����*�6�C�M�O�Y�U�O�F�C������������������������������������������������������������������������������������
����'�)�6�B�E�B�@�8�6�3�)�!��ѿ˿Ŀ����������Ͽݿ�����������ݿѿѿʿĿ����������������Ŀƿѿտڿݿڿѿ���������������������������������������ٻ����z��������������������������������������������	���.�;�G�T�`�h�l�`�T�.�"��нɽͽнؽݽ������ݽннннннн������~�F�5�B�g�������������������������������������������(�1�(����ìëàÛÙÞàìù����������úùìììì������#�(�+�5�=�A�B�A�5�/�(�����������ݿӿѿĿĿɿѿݿ޿�����������ݿԿѿۿݿ߿�����ݿݿݿݿݿݿݿݿݿ���
�����"�#������������Ɓ�{�y�ƁƎƖƚƧ������������ƧƠƚƎƁ��������������������������#���������������޺������!�-�2�)�'�!��������������'�@�Y�f�q������f�Y�@�����/�,�"�����"�/�1�;�H�T�H�H�;�/�/�/�/ììàÓÓÓàì÷ùüùìììììììì�h�]�[�P�P�[�b�hāčĦĭıĦĚĘčā�t�h�׾Ҿ;ʾƾʾվ׾پ�������ܾ׾׾׾����y�s�r�w�����������������������������������������������������������������������@�=�@�L�T�Y�e�h�e�c�Y�L�@�@�@�@�@�@�@�@����z�m�f�c�c�h�m�z������������������������������������������� ��	������������������������)�.�6�9�<�?�8�)�����ǈǇ�|�{�v�{�ǈǔǗǡǫǭǡǠǔǈǈǈǈ���������������������������������������ؼ������� �4�@�M�Y�e�a�Y�M�I�@�4��������������B�[�hāĚĲĵĬā�h�O�6��Ŀ����������ĿѿܿܿӿѿĿĿĿĿĿĿĿĽн����������~���������Ľнٽٽ߽�ٽӽ��H�E�;�3�/�.�/�;�E�H�I�K�H�H�H�H�H�H�H�H�ɺȺ����������������ɺ˺ֺ�����ֺɾ���������������(�4�=�A�J�F�(������ ���!�-�:�F�l�u���������x�l�-�!��t�g�[�P�[�]�g�q�t£�t�a�T�M�K�T�Y�a�g�m�p���������������z�m�aE�E�E�E�E�E�E�E�E�E�FF$F(F$FFFE�E�E湝�����������ùܹ��������ܹѹ���������������'�3�L�^�r�������~�r�e�L�3�'�������������Ľݽ��������нĽ��������лȻŻû»ûϻллӻܻ�ݻܻллллл�����������������������������������������������ùϹܹ�����ܹϹù���������������*�6�6�=�6�6�*���������ŹŪŝŜšţŭŹ�������������������߽.�!�������.�:�G�S�`�g�`�V�I�:�7�.���������������������������������������z�o�p�p�y�zÇÌÓÕÞàäàÓÇ�z�z�z�z�U�U�H�E�B�H�U�\�`�U�U�U�U�U�U�U�U�U�U�U�
����������
���"���
�
�
�
�
�
�
�
��ܻӻл����»лܻ�������
������@�4�'��� �'�8�@�M�Y�f�j�o�o�p�o�f�Y�@ECEBE7ECEDEPEUE\E]E\EVEPECECECECECECECEC v ^ > v 4 ; =  o R 0 j Q I , X % Y   M F @ M A 3 / t D / W / B L P ` ( / z V N X D F J w t > 4 K  `  L " ' U C � : 9 �   F 8 * V [ ` l X p = 8 T B 0 W > O ? O G 1    ^  �  �  (  �    5  �  o  s  [  �  �  ~    �  m  F  �    �  �  �  y  u  p    �  '  �  6  �  �  ^  �  �  �  _  m  p  �  G  �    �  )  �  �  �  ?  U  �  �  �  �  �  �  �  �  q  �  4  �  d  t  �  }  #    O  �  �    �  �    m    U  �  �  t  O<�9X<#�
��t����
��o��`B�u����o��`B�o�o�49X�0 ż�/����C����ͼě�������/�t����o���
������������ͼ�9X��j��w��/�u��`B�'�h��h����h�\)�]/�P�`��C���P��%���L�ͽ,1���P�`�P�`��O߽e`B�,1�D������L�ͽ���0 Žy�#�y�#���w�����\)���
���罝�-��\)�]/�q�����+��O߽��T��o��hs��O߽����9X��h����B+cZBaBߌBe/B B!iB�"B�Bm�B+gB4ѹB v9B�ABJ�B�BB �mBSPB��A��mB&�B�bB0�jB ҷBq�B �B��BvB�B2�B��B�WB�'B&�|B)�gB�EA��3B~B��B]*A���B�JB�B!�mA���B.�BsXB�B��B�UB�%B��BlZB��B�-B0B)�B;�B��B%�B|�B#�&B&��B%��BeDB&_B�gB�B_�BkgB&H	B��B7yB��B
L�B?�B�LB�B`�B��B�pBѫB2CB+��B��B�BFB:�B �3B��B�ABQ�B��B4�=B �?B��B@	B2DBO�B ��B?]B��A��B�2B/.B0D�B ��BJ�B?�BA�B��BhsB?OB��BBB��B&�^B)��B�}A�׆B�B�Bv�A���B?�B��B"wA��B<�B�'B7�B��B�$B�|B��Bq�B@�B>BX�B)��BĕB��B%H'B�B#�iB&��B%A�B@�B��BDB�6BB�BF�B&:�B�CB@B��B
J�B?<B��B@'B�gB� B�GB41B?�A��A�,�A��A���A0��AŰ8A:�RA��A�@B-�AP9A�3�A?�C�9�A�H�A��AB�A���A�*"A�:�A��Ab[DA\>HB WAI��AJE�Aֈ!A|aAx�dB}�@�� Aa�A*�9A��A���A���A���A~�!A|�A��*B��A��@ZD@�#1A��uA�#�A��BAS�cA�� A��F?Φ�A��iA�=<A�	TB%TA��@�tAA�ѻAy�<A"��A�b@1�jA6
�@�ƦA��EA�� C���>%%?�!,A'��@�\Bz/>��A�$�A�1A�A�AɀNA�8A��[@�	�@ԫ�C��6A�~A��oA�V�A��A/nKA�l'A<��A�n�Aĝ�B��AO�A�pA>��C�B`A̐�AƊPAD��A�|A�M�A���A�zAc BA[B <�AIxAJ��A��A~DAy&B��@��A_�A+�A���A���A�MyA��A~G�A}A��B�%A���@Tuo@��A���A�p7A܈jATcRA�yA���?�%DA��A�{�A��B?�A�}�@���A���AyOA!��A��@2OA6�{@|�A���A�G�C���>>�s?��dA'j�@��B�'>2?A��cA��dA�A�yAɃ�A�N�A�z�@��@�C��N                                        	   %                                       
                     '                     	         8                   	            !            .                  #            !                              	            0      
                  #         '                                    +                              #      5                        /      %                                          7                  %      '         '                                             
                           %                                    )                              !      5                        /                                                1                  #      '         '                                          Nf�N�gOYOP�&NѝBN�1kO�	�N���N&mO��NA�@O�@gN���O&ON�<ND��O�8�N�|�O3�@OM��OEx�P٘OdХO=bN_�`N]�N���O_��N���NaKgN�O���N{��PcTRNlwsN���Ny��N���N /oN:��O�`P8ŗO)CvO�B�N�G�M�ºN�2N���OB�O�N&�Oe��On��OۥN���N�� O7^�P�'N�{�OTL�M�
�O&�NЫ�OിO�O�uZO�IO&O�1}O!N,ŖN{@�N��NF�O�:�O�GN7��N���N WtNO(O/��O��yNCE�  X    e    c  �  �  g    �  8  �  �  )  `    A  A  �  �  k  �  ~  '  �  �  �  �  k  �  �    �  >  �    �  y    q  �  �  j  �  �    �  ,  �  w  J  �  N  �  �  L  �  �  &  �  �  �  g  �  *  �  �  -  �  �  �  �  )  �  k  �  l  �     �    	  [<���<e`B;D��;ě�;ě�;�o�o���
;D��$�  ;o�o�D����`B�#�
�o��`B�D���#�
�49X�49X�T����t���o�u�u��C���t���t���t���t���j��9X��9X��j�������ͼě��ě����ͼ�`B��/��`B����h�o�,1�C��C��C��C��\)�\)�<j��P��P��P�,1�#�
�<j�''<j�@��D���D���P�`�e`B�P�`�ixսP�`�T����C��m�h�e`B�ixսu��%��%��o��O߽�hs��E�������������������������������������������������������������������������������������������������������������������������@BN`glqqj`[UC@<<??=@��
!#''&#
������nnpz����znnnnnnnnnnn�����������������������������������tz������������}zsoqt����� � ����������������������������������
#(/.#"
�������7BOT[\[RONB:77777777��������������������������������������#/<HINQQOIH</+#TVZ`bmz}���~zmja\WTT��� 	�������)Ohtyzqqkili[B6$#"*6COU]cif\OC6*����������������z|���������������������������������������

"'#
����/58BMOT[gostg[NG<50/���������

��������� 	


�����������[[httutkh`^[[[[[[[[[ )5BN[gig\Zfla[NB$  #'/;5/#	#b{�����sU<0
	����������������������������������������;;?HTZ`YTH@;;;;;;;;;���������������������������������������������������������TU[agmszzzzwpmaWWYTT���5BQQB94�����������
����������������������������������� ����������������������������������������������������GHUZafnqnja]UPNHGGGG'6:BGB63)
U[itz�����}vthb[XQSU�������������������������
#&)*(#
����	#/<HUY\]UH</#
	����������������������������������������DO[hpqnh[WOGDDDDDDDDy{������������}{xxy�)58)%-:>5)����_adntz��zna`________�
#06<A@><80#
����7BIOZ[\_[UOB77777777������

	�����4<BILTUVWVUMI?<84344
0DQWTUTI<#
	
�������������������������������������
#+/24799/-#	CHUWamruvolaURJHEA>C��������������������)),+)&(0<CIKUUUIB<0/((((((��������������������MO[hihf][YONMMMMMMMM45BBBNLKB95444444444Wddgt���������ytg`TW���������������������������������)2/)!������BBJN[d`[NBBBBBBBBBBB�����������������������������������������������
�������:<HUUVULHG<4::::::::�`�_�]�`�l�p�t�m�l�`�`�`�`�`�`�`�`�`�`�`�s�l�g�^�\�d�g�r�s�}�~�y�s�s�s�s�s�s�s�s�
����������
�
��!�/�6�<�?�?�<�/�#��
�������������������	��"�.�"���������׽��������������������������U�P�H�A�<�;�<�D�H�U�a�g�g�l�c�a�U�U�U�U����� �(�A�M�f�s�u�s�f�Z�M�A�4�(���N�G�A�:�5�A�D�N�Z�g�n�s�n�g�`�Z�N�N�N�N�U�T�H�D�C�H�U�X�X�U�U�U�U�U�U�U�U�U�U�U��Ƽƶ���������$�0�5�7�0�$����������̾��������������ʾԾԾξʾ���������������ìàÓÇ�z�v�m�m�zÇàì������������ùì�M�L�A�?�9�A�M�Z�f�s�w�z�s�f�\�Z�M�M�M�MD�D�D�D�D�D�D�D�D�D�EEEEEED�D�D�D�àÞÜØ×Üàìù����������������ùìà�a�V�[�a�m�n�u�z�{�z�s�n�a�a�a�a�a�a�a�a�M�E�J�I�C�H�M�Z�s����������������f�Z�M��t�r�t ¦¦ª¦���������������������������������������t�h�[�O�O�S�^�h�tčĚĦīĪĦĠĚčā�t�����������������+�5�B�I�B�5�1������.����������	�"�;�G�`�m�y���~�p�m�`�G�.�	�����������	��"�(�+�-�/�.�(�"��	�*�"����	���*�6�C�L�O�S�P�O�C�6�*�*������������������������������������������������������������������������������������
����'�)�6�B�E�B�@�8�6�3�)�!��ѿ˿Ŀ����������Ͽݿ�����������ݿѿѿʿĿ����������������Ŀƿѿտڿݿڿѿ���������������������������������������ٻ����z������������������������������������	������� ��	��.�;�G�T�`�e�f�[�.�"��нɽͽнؽݽ������ݽннннннн������~�F�5�B�g������������������������������������������$��������àÝÛàãìù����������ùìàààààà�(�(�����(�5�:�>�5�(�(�(�(�(�(�(�(�(�������ݿӿѿĿĿɿѿݿ޿�����������ݿԿѿۿݿ߿�����ݿݿݿݿݿݿݿݿݿ���
�����"�#������������ƚƎƁ�ƁƂƎƙƚƧƴ����������ƳƧƝƚ��������������������������#���������������ߺ�������!�+�(�&�!��������
����'�@�M�Y�f�r�{�{�r�f�Y�@�4���/�,�"�����"�/�1�;�H�T�H�H�;�/�/�/�/ììàÓÓÓàì÷ùüùìììììììì�h�g�\�[�Z�[�h�tāčėĖčāā�t�h�h�h�h�׾Ҿ;ʾƾʾվ׾پ�������ܾ׾׾׾����y�s�r�w�����������������������������������������������������������������������@�=�@�L�T�Y�e�h�e�c�Y�L�@�@�@�@�@�@�@�@����z�m�f�c�c�h�m�z������������������������������������������� ��	��������������������������)�/�4�5�/�)���ǈǇ�|�{�v�{�ǈǔǗǡǫǭǡǠǔǈǈǈǈ���������������������������������������ؼ������� �4�@�M�Y�e�a�Y�M�I�@�4���	������������B�hāĚīİĦā�h�O�6��Ŀ����������ĿѿܿܿӿѿĿĿĿĿĿĿĿĽ��������������������Ľͽ˽ʽӽнȽĽ����H�E�;�3�/�.�/�;�E�H�I�K�H�H�H�H�H�H�H�H�ɺȺ����������������ɺ˺ֺ�����ֺɾ���������(�4�6�A�E�A�?�4�(�������!�-�:�F�S�l�����~�~�t�l�_�F�-���t�p�g�[�c�g�t�y�t�t�a�T�M�K�T�Y�a�g�m�p���������������z�m�aE�E�E�E�E�E�E�E�E�E�FF$F(F$FFFE�E�E湝�����������ùϹܹ���߹ܹϹ̹ù�������������'�3�L�^�r�������~�r�e�L�3�'�����������������Ľнֽݽ���ݽսнĽ����лȻŻû»ûϻллӻܻ�ݻܻллллл�����������������������������������ù������ùʹϹܹ߹ܹѹϹùùùùùùù��������*�6�;�6�.�*������������ŹŪŝŜšţŭŹ�������������������߽.�!�������.�:�G�S�`�g�`�V�I�:�7�.���������������������������������������z�o�p�p�y�zÇÌÓÕÞàäàÓÇ�z�z�z�z�U�U�H�E�B�H�U�\�`�U�U�U�U�U�U�U�U�U�U�U���
���������
���!�����������ܻӻл����»лܻ�������
������M�@�4�'��"�,�9�@�M�Y�f�j�n�n�o�n�f�Y�MECEBE7ECEDEPEUE\E]E\EVEPECECECECECECECEC v ^ / v 4 8 G  o L 0 m Q I 3 O % V  G F = < ? 3 / t D / W / C L P O " ( z V N P D 0 9 w t * 4 K  `  L  ' U C � : A �   0 2  V [ N l 8 p = . < B 0 W > O = O E 1    ^  �  E  (  �  �      o  #  [  f  �  ~  �  }  m  �    �  �  �  �  G  u  p    �  '  �  6  �  �  ^  z  �    _  m  p  V  G  h  <  �  )  �  �  �  ?  U  �  �  =  �  �  �  �  �  �  �  4  �  �    �  }  �    2  �  �  =  Y  �    m    U  s  �  N  O  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  X  W  U  T  R  R  X  ^  d  j  p  v  {  �  �  �  �    I  |        �  �  �  �  �  �  �  �  �  �  �  �  w  m  c  Y  O  �    C  U  `  e  c  X  H  /    �  �  j     s  �  A  �  �        �  �  �  �  �  �  t  V  3    �  �  y  P  2    �  c  ]  W  M  D  8  ,         �  �  �  �  �  �  �  }  Y  3  �  �  �  �  �  �  �  �  �  �  �  �  �  e  D  #    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  R  3    �  �  �  �  �  �    1  G  X  d  f  a  V  E  -    �  �  P    �  k  1      
                   !        
          �  �  �  �  �  �  �  �  s  N  %  �  �  �  J    �  �  B  �  8  7  5  4  2  0  +  &  !           �   �   �   �   �   �   �  �  �  �  �  �  �  {  `  4  �  �  S  �  �    �  �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  d  N  6      )      �  �  �  �  �  ^    �  h  �  �  �  p  �  +  {  �  A  W  ]  _  Z  S  H  7       �  �  �  �  z  V  4  	  �  �  �  �    
    �  �  �  �  �  �  �  �  �  �  �  m  f  6  �  A  >  ;  5  ,  !      �  �  �  �  u  ;  �  �  �  I  +      %  0  :  ?  ;  +    �  �  �  �  a  4    �  �  z  �  �  �  �  �  �  �  �  �  �  �  y  ]  @  #    �  �  �  X  !  �  |  �  �  x  k  Z  G  3    �  �  �  �  o  6  �  �  V    �  k  _  R  F  @  B  A  8  '    �  �  �  �  �  �  �  e  D  $  �  �  �  �  �  �  �  �  �  �  k  ^  U  @    �  �  {     �  Z  m  w  }  }  |  |  y  r  i  X  >    �  �  �  f    |   �    &  #      	    �  �  �  �  �  �  g  I  /      	    �  �  �  �  �  �  x  m  b  V  J  >  &    �  �  �  v  N  &  �  �  �  �  �  �  �  �  �  �  �  �  r  e  W  =        �   �  �  �  �  �  �  �  �  �  j  _  ]  R  <  '        �    6  �  �  �  �  �  �  s  `  H  /    �  �  �  �  n  B    �  �  k  `  U  J  ?  3  '      �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  r  h  W  E  4  "    �  �  �  �  �  s  T  5    �  �  �  �  �  �  �  �  �  �  �  x  p  h  a  Z  S  M  G  A  �          �  �  �  �  ~  N    �  �  v  E  �  �  �  u  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  o  `  P  ?  /    >         �  �  �  s  D    �  �  �  �  p  R  *  �  s   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  h  \  P  �  �  
      	  �  �  �  �  �  \  3    �  �  v  A    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  i  U  @  *    y  m  a  U  I  <  /  #            �  �  �  �  �  �  �        �  �  �  �  �  �  �  {  ^  @  !    �  �  �  �  g  q  l  g  c  ^  X  M  B  7  -  !      �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  a  N  ;  '    �  �  �  �  �  e  =  �  �  �  �    b  C    �  �  �  e  D  R  I  9    $  �  y  @  g  Z  K  ?  7  %  	  �  �  �  S    �  �  c  #  �  p  �  	  �  �  �  �  �  �  �  �  b  0  �  �  h  �  }  �  Z  �    �  �  �  �  �  �  �  �  �  ~  i  T  6    �  �  �  h  5           �  �  �  �  �  �  �  �  t  _  I  2      �  �  �  �  �  :  s  �  �  �  �  �  �  {  W  (  �  �  l  $  �  �  �  ,  !        �  �  �  �  �  �  �  �  �  w  V  /     �   �  �  �  r  \  F  .    �  �  �  �  y  X  7    �  �  h  �  S  w  t  q  l  g  a  Y  Q  C  6  %    �  �  �  �  �  h  D     J  O  U  Z  _  b  ]  X  S  N  F  :  .  "    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  N  .    �  �  �  �  L  �  N  F  8  $    �  �  �  �  o  J  (    �  �  �  A    �  �  J  b  y  �  �  �  �  �  �  b  8  �  �  A  �  -  �  �  $  $  �  �  o  Y  >    �  �  �  �  ^  4    �  �  B  �  �  S  �  L  B  8  /  %        �  �  �  �  �  �  �  �  a  6     �  �  �  �  �  {  h  Z  M  7  !    �  �  �  �  �  o  R  3    �  �  �  �    E  �  �  �  v    �  �  `  g  �  �  
      &  &  %  !          �  �  �  �  �  d  6    �  �  p  ;  �  �  �  �  �  �  �  �  �  �  �  �  z  ^  5    �  �  g  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  K    �  �  ~  D    ~  C  A  E  c  f  f  `  S  <    �  �  �  c    �  `  �  �  "  �  �  �  �  �  �  �  �  �  �  u  K    �  q  �  �  9  2  <       '  )  %         �  �  �  �  z  N    �  w    �  +  �  �  �  �  r  L  "  �  �  �  a  ,  �  �  �  H    �  _   �  �  �  �  o  ?    �  �  �  �  �  I    �  �  t  0  �  �  (  (  *  *  ,  #    �  �  �  r  I    �  �  y  )  �  Z  �  z  �  �  �  ~  W  -    �  �  �  U  T  O  N  I  (  �  �  �  u  �  �  }    �  �  �  �  �  m  O  *    �  �  �  k  l  z  b  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  u  l  c  Z  Q  �  u  j  `  Z  T  K  @  5  .  (  !      �  �  �  �  �  �  N  �  �  �  �        '  )  "    �  �  �  _  *  �  �  g  ^  c  m  �  �  �  �  y  Z  7    �  �  �  Z    �  �  +   �  k  [  K  9  '    �  �  �  �    ^  C  3      �  �  �  K  �  �  �  n  K  %  �  �  �  �  \  )  �  �  v  C    �  )   �  l  l  k  j  i  h  e  b  _  \  X  T  P  L  G  8  %    �  �  �  �  ~  e  G  )    �  �  _  7  .  )  ?  V  F  *    �  �           �  �  �  �  �  �  �  �  �  �  �  z  f  S  @  -  �  �  �  �  �  �  �  �  �  �  �  s  X  <    �  �  �  �  ]    �  �  x  ;  �  �  r  ,  �  �  �  q  B    �  �  Y    ^  	  	  	  �  �  �  G    �  r  .  �  �  3  �  q    �  �  p  [  .     �  �  �  a  =    �  �  �  q  D    �  �  �  �  �
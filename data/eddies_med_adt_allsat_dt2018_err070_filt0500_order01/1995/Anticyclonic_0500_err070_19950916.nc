CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�bM���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�w�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��P   max       =�v�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @E������     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v|��
=p     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @N            |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�}        max       @��@          �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��j   max       >ě�      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�+�   max       B-��      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��;   max       B,�/      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C��h      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?0��   max       C���      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         }      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          ?      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          ?      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�w�   max       P���      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�
=p��   max       ?�ڹ�Y�      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��P   max       >_;d      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @E������     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v|��
=p     	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @N            |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�}        max       @��@          �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A6   max         A6      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?q4�J�   max       ?�e��O       T�            <      f     }   E               �   (      1   "   	      
         8         "         �   F   B                            
   ;   	             %               �                        6   N�"�Nn�fNp(�P?�RO=�P��NGk�P�{IP�єO(Z�N��3M�w�Ni�P���O��N�P7�O��\O&p�O���N�?`N��O֑P'�oNT#N�O5u=O[6rN���O���P���P7�,O�&.NG=�N�ΡOO��^N��N�QrO%MlO@�O�R�Ni	zO!)<O���O	<NOȋtO*O[șOH�N��Po�N�ߴNُO4��OP�qOX��Oj�N��yP��O�k1��P������j�#�
�ě�:�o;D��;�o;��
;��
;��
;��
;�`B<D��<D��<T��<e`B<u<�9X<ě�<ě�<���<���<���<���<�/<�`B<�h<�<��<��=o=+=C�=C�=\)=��=��='�='�='�='�=,1=0 �=0 �=D��=L��=P�`=T��=��=�7L=�O�=�t�=���=���=���=��=� �=�9X=�Q�=�v�WO[gt�����thg[WWWWWW�������������������������������������������)0<GUVPH0#
�����ggit�������������tmg���N[dfZB5!������aijaUNIIUaaaaaaaaaaa)Bg������t[B]]cm������������zng]z����������������{z����������������������������������������������
�����������;;LWh������������bC; &0<IU]cecUIA<0#����������������������������������������)5BGNONF5)	)+>=BCKB:5)���)5BNNA5)����������� �����������

����������$&6BO]beb`[O6*&�������������~vx�������~~~~~~~~~~-/<HNKH<7/----------��������������������������������������������

����������������

�����xx���)/$������x��������������������������)*%$���\]ht~{tmh\\\\\\\\\\
#.000(#
,+./5<HY[ZUNFA<;41/,$/<HNUWZ[_^\TH</#]Y_anyz}~zxnfa]]]]]]^\^_acmz}}|ztmda^^^^������������������������
#*)#
������ox}���������������zo��


	 �������������������������������� #
�������� )25=BIJB5,)'" ������(+)��������������������������)5BEJLOONHB50)!5//6;BHTamnpomfaTH;5�}�������������������������
'02.
����##"#'0<B@A<;70)#####xwz��������zxxxxxxxx���������  ���������)++./)!%')1588:95)QNTUaanz����~zunaUQQ#+-*#
		
#######)#)��������)-)#.7<DGHF</#�a�n�u�v�z�w�n�a�U�U�Q�U�U�`�a�a�a�a�a�a�'�0�3�@�@�@�>�9�3�-�'���$�'�'�'�'�'�'�=�B�I�J�I�E�=�0�$��$�*�0�1�=�=�=�=�=�=�������ûлܻۻ������l�_�J�C�G�C�S�l���������������������������������������������I�nńņŁ�}�n�b�<�!�����ĿĽ�������#�I�)���
����)�,�)�)�)�)�)�)�)�)�)�)�)�)�B�[čĞĬįĠĒā�h�6�)���������)�����������������������g�N�/�+�1�A�Z�s��FF$F,F1F;F9F1F+FFE�E�E�E�E�E�E�E�FF�/�<�H�T�T�H�B�<�2�/�*�#�"� �#�(�/�/�/�/²¿��������¿²¯¬²²²²²²²²²²�O�[�h�t�v�|�t�h�[�S�O�K�O�O�O�O�O�O�O�O�'�@�r�����Ƽ�������6�'�����ػһ���'�Y�r���������������������^�P�M�F�@�D�Y�������������������������������������������������,�.�)����������îòììò�ſ������ĿѿͿƿ����������y�n�n�k�z��������"�/�;�H�T�X�V�T�H�;�0�/�"���������$�G�O�R�V�U�W�N�A�(������������������������������������������������������������������������񾱾ʾԾ������ʾ������s�l�k�z�����������6�;�L�\�g�\�C�������ŷůŰ��������5�A�N�Q�Q�N�A�7�5�2�5�5�5�5�5�5�5�5�5�5¤Óàóù��������ùìàÓÒÇÂÁÄÇÈÓ�5�A�N�Z�g�i�r�t�m�g�Z�N�A�5�,�(�!�"�(�5�[�_�g�o�t�v�t�g�[�N�B�9�>�B�N�V�[�[�[�[DoD{D�D�D�D�D�D�D�D�D�D�D�D{DhDVDVDTDbDo������*�D�F�>�/����������������������׽���4�A�I�J�3�1�(���н����������Ľݽ��"�;�T�`�y�����q�`�T�;�.�"��	�����"�-�:�=�B�:�-�!��!�%�-�-�-�-�-�-�-�-�-�-������!�(�,�!����� ������������������������
�������������ù÷ù�������޿����0�@�K�N�\�Z�N�5�(������������g�s���������}�s�g�]�Z�T�Z�b�g�g�g�g�g�gŔŠŭŹſ��������ŹŭŧŠŘŔŎŔŔŔŔ�H�U�a�g�e�a�]�U�H�<�/�#����#�/�2�<�H�/�;�A�?�?�=�;�7�/�"�������"�$�/�/�����ɺֺݺ� ������ֺɺ��������������m�t�y���y�v�m�`�Z�T�S�T�`�e�m�m�m�m�m�m�G�T�`�m�z�x�y�~�y�m�a�`�_�T�Q�G�?�=�C�G�.�;�G�N�N�G�"�����Ѿʾ��Ҿ׾���	��.�A�M�Z�Z�W�M�J�A�=�4�(� ������(�4�A�y�������������������y�l�f�f�Z�Y�\�`�l�y�	��������	��������������������	�0�=�I�S�S�N�B�=�0�$���	���������	��0čĚĦĳķĿ��������ĿĳĦĤĚĕČćĆč�@�M�Y�f�h�r�����r�r�f�Y�M�E�@�9�=�@�@E7ECE\E�E�E�E�E�E�E�E�E�EiE\EHE@E4E/E+E7���ɺֺں����ֺɺ����������������������ʼмʼǼü������������������������������ûлܻ���׻лû��������������������S�l�x���������x�l�X�S�F�:�-�!�!�-�:�E�S�N�[�g¦²¸¦¢�t�g�V�@�<�G�N���(�)�4�5�?�@�4�(�����������ǭǡǔǈǁǅǈǓǔǡǭǮǱǭǭǭǭǭǭǭ�@�3�&�����͹����Ϲ��3�J�e�k�p�j�Y�@�����6�C�O�Y�O�C�6�-���������������� U b I , O B A  9 G @ F C 5 6 : 3 $ 1 W > x C L 2 Y   . ! ? I . H : < Y $ ] T S H 0 / ^ Y K 7 R : F B C W 6 p q 1 4 r V      �  �  /  �  .  R  �  S  �  �    �  �  p    *  }  p  �  �  y     V  (  D  |  �  �  �  :  W  �  m  �  [  j  �  �  �  [  �  y  k  �  c    r  �  �    �  �  >  �     5  O  �  "  s��j��t��#�
=H�9;ě�=��`<t�>ě�=��<�<���;�`B<���>�-=P�`<�h=}�=D��<��=#�
=C�<�`B=T��=��<�`B<�=y�#=u='�>)��=��=Ƨ�=��=#�
=,1=}�=�O�=49X=<j=�%=P�`=���=L��=ix�=���=e`B=�-=��=�hs=�v�=�1>I�^=��=��-=Ƨ�=�^5=���=�/=��>t�=�B	�ZB �BTLB$�HB
��B�#B�JBa�B ��Bo�B�*B l�B�B�NB&SZBtB�'B��B�YBT�B�AB�B<ByB
^cB��B!��B��B�DB�JB�^B!�wB�B��B$ՔB�{B��B��A�B=B��B��B��B��BvABDB��B-��B)�BBA�+�B*�B��B%�B��B��ByBn�B��B\�BE�B��B	�mB ԒBAB$�wB
�~Bd%B? BC�B ��B?�B�bB )�B>]B��B&AwB��B��BƨB�uB@�BB�BA B@QB�%B
A�B�IB"<QB�.BE�B�7B��B!��B�B-�B$��B�RB�MB�AA�qB��B9�B�9B��B�;B��B��B,�/B@�B=�A��;B?�B�9B%��B��B@�B+5B��B��BE"B��B�A� ?�9�B
g^@�zQA�l`A��A�R�A�_�A��8C��hA�FIA�^A�M@�8@��A�UmA���ArƝA��FA���A��A��XAL)DA��A���A�kA��lA��A�L	C���A�,A,��Ac�@v;A
13AѹA�#OA�)A��AívA�PO@3�-Aj|�Ah��A[FVA8�ZA�7A��pB	�iA�@؏ C��!@5�_@��:@��C@��#A�ۋA5t�Bz??���A�WAƋ?�q�B
C�@�PnA��^A�{�A�kLAڃ�A��C���A��VA��&Aۄ�@�
_@���AЁ�A�Ar��A�:\A�1�A�}�A�urAL�TA���A�|{A��A�yA���A�y�C���A�w�A,��Adqv@u��A	3Aњ�A���A��8A�~�AāaA���@3F|Aj�@AgowA[tA5�A�A��~B
;CA���@�$C���@:ׁ@�9�@���@�!A���A5-�B�?0��A�w      	      <      f     }   E               �   (      1   "   
               9         "          �   G   B   !      	                      ;   	      !   	   &               �                        7               3      9   
   9   3               ?         +                  #   -                     ?   /   !                                    !      #               +                        /                     %   
      )               -         '                     #                     ?                                          !      #                                       /   N�"�Nn�fNp(�O��!O�rP >%NGk�O�ZP�;O(Z�N���M�w�Ni�P$aiO�ݠN�CEP&�Oi;cO&p�O���N�{N��Oi��O��NT#N�O�O�N���O2݄P���OO�dN��7NG=�N�ΡN���OA<�N��N�QrO%MlO@�Oy�2Ni	zOŀO�2O	<NOȋtN�J�O[șOH�N��<O4i9N�ߴNُO4��OP�qOX��Oj�N��yP��O�k1  �  �  �  �  H  	  $  .    q  6  I  �  
�  �  �  %  �     �  �  P  �    �  a    1  .  z  /  �  �  �  �  y    0  �  R  �  	�  �  �  �  3      �  (  �    b  ]  �  �  *  �    
e  D��P������j<T����o=0 �;D��>_;d<�`B;��
<o;��
;�`B=ix�<�t�<�t�<�C�<�j<�9X<ě�<���<���=C�=�w<���<�/<��=�P<�=�1<��=��=T��=C�=C�=49X=0 �=��='�='�='�=T��=,1=8Q�=8Q�=D��=L��=e`B=T��=��=�C�>o=�t�=���=���=���=��=� �=�9X=�Q�=�v�WO[gt�����thg[WWWWWW��������������������������������������������
#0<@BA=60#
���pmkqt�������������tp�������+22+%���aijaUNIIUaaaaaaaaaaa/-.25BN[gmuxwrg[NB5/lmz�������������zsplz����������������{z����������������������������������������������
�����������NRU[h�����������j[ON  $)0<IX_a_UKI20#����������������������������������������

)5<BCHHB5)
)+>=BCKB:5)���)5BNNA5)������������������������

����������&%(06BOT\]^[TOB=62/&��������� �������~vx�������~~~~~~~~~~-/<HNKH<7/----------��������������������������������������������

����������������

	�����xx���)/$������x�������������������������	 ���\]ht~{tmh\\\\\\\\\\
#.000(#
//2;<?HRSMH<3///////""'/7<HUWZYWUPH</#"]Y_anyz}~zxnfa]]]]]]^\^_acmz}}|ztmda^^^^������������������������
#*)#
������}������������������


	 �������������������������������������"
��� )25=BIJB5,)'" ������(+)��������������������������)5BEJLOONHB50)!5//6;BHTamnpomfaTH;5�~������������������������
## 
����##"#'0<B@A<;70)#####xwz��������zxxxxxxxx���������  ���������)++./)!%')1588:95)QNTUaanz����~zunaUQQ#+-*#
		
#######)#)��������)-)#.7<DGHF</#�a�n�u�v�z�w�n�a�U�U�Q�U�U�`�a�a�a�a�a�a�'�0�3�@�@�@�>�9�3�-�'���$�'�'�'�'�'�'�=�B�I�J�I�E�=�0�$��$�*�0�1�=�=�=�=�=�=�����������������������x�l�e�^�\�[�_�m���������������������������������������������
��0�<�L�Z�b�^�U�I�0��
�������������)���
����)�,�)�)�)�)�)�)�)�)�)�)�)�O�[�h�t�yăąĂ�z�t�h�[�O�B�5�/�.�2�@�O�������������������g�Z�N�I�C�@�E�V�g�s��FF$F,F1F;F9F1F+FFE�E�E�E�E�E�E�E�FF�/�<�H�O�N�H�@�<�/�&�#�,�/�/�/�/�/�/�/�/²¿��������¿²¯¬²²²²²²²²²²�O�[�h�t�v�|�t�h�[�S�O�K�O�O�O�O�O�O�O�O�4�M�f�����������r�M�@�4�'�������'�4�Y�f�r�������������������g�Y�W�M�K�P�Y����������������������������������������ù���������*�+�'����������ñòõðù�����������Ŀſ��������������z�y�v�y������"�/�;�H�T�X�V�T�H�;�0�/�"���������$�G�O�R�V�U�W�N�A�(�����������������������������������������������������������������������������񾱾��ʾ׾ݾܾҾʾ���������|�|��������������*�6�?�A�>�*���������Ÿź���������5�A�N�Q�Q�N�A�7�5�2�5�5�5�5�5�5�5�5�5�5¤Óàëìù����������ùìàÓÇÄÃÇÑÓ�N�Z�d�g�h�g�b�Z�N�A�5�3�(�&�(�(�5�A�F�N�[�_�g�o�t�v�t�g�[�N�B�9�>�B�N�V�[�[�[�[D{D�D�D�D�D�D�D�D�D�D�D�D�D}D{DlDmDoDuD{������*�D�F�>�/����������������������׽ݽ�������������ݽнĽ����ǽнݿ.�;�G�R�T�`�b�`�V�T�G�;�/�.�"�!�"�%�.�.�-�:�=�B�:�-�!��!�%�-�-�-�-�-�-�-�-�-�-������!�(�,�!����� �������������������������������������������������������+�5�;�A�?�5�(�������������g�s���������}�s�g�]�Z�T�Z�b�g�g�g�g�g�gŔŠŭŹſ��������ŹŭŧŠŘŔŎŔŔŔŔ�H�U�a�g�e�a�]�U�H�<�/�#����#�/�2�<�H�/�;�A�?�?�=�;�7�/�"�������"�$�/�/�������ɺ���������ֺɺ��������������m�t�y���y�v�m�`�Z�T�S�T�`�e�m�m�m�m�m�m�G�T�`�l�m�w�v�y�z�y�m�`�W�T�G�@�B�D�G�G�׾���	�"�.�;�G�K�L�G�"�����Ӿʾþʾ׾A�M�Z�Z�W�M�J�A�=�4�(� ������(�4�A�y�������������������y�l�f�f�Z�Y�\�`�l�y�������	���	�	�������������������������0�=�I�S�S�N�B�=�0�$���	���������	��0čĚĦĳķĿ��������ĿĳĦĤĚĕČćĆč�@�M�Y�f�f�q�p�f�Y�M�F�@�:�=�@�@�@�@�@�@EuE�E�E�E�E�E�E�E�E�E�EuEiE_E\EZE\E\EiEu���ɺֺں����ֺɺ����������������������ʼмʼǼü������������������������������ûлܻ���׻лû��������������������S�l�x���������x�l�X�S�F�:�-�!�!�-�:�E�S�N�[�g¦²¸¦¢�t�g�V�@�<�G�N���(�)�4�5�?�@�4�(�����������ǭǡǔǈǁǅǈǓǔǡǭǮǱǭǭǭǭǭǭǭ�@�3�&�����͹����Ϲ��3�J�e�k�p�j�Y�@�����6�C�O�Y�O�C�6�-���������������� U b I , K C A  + G C F C : 8 4 )  1 W ? x 8 D 2 Y   .  ? + > H :  D $ ] T S @ 0 / _ Y K - R : 0  C W 6 p q 1 4 r V      �  �  �  \  O  R  3  �  �  �    �  �    �  �  �  p  �  �  y  �  !  (  D  P  %  �  s  :  �  �  m  �  �  �  �  �  �  [  
  y  2  �  c    �  �  �  �  u  �  >  �     5  O  �  "  s  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  A6  �  �  �  �  �  s  T  3    �  �  �  }  O    �  �  j  *  �  �  �  �  �  �  �  u  g  U  B  -    �  �  �  �  ]    �  q  �  �  t  \  L  C  @  D  S  k  �  �  �  $  e  �    �  �  _  �    5  U  s  �  �  �  �  �  |  Z  /     �  �  /  �  ^  =  D  G  H  H  F  C  <  5  .  +  &      	    �  �  �  V  �  S  �  �  :  h  �  �  	  	  	  �  �  V  �  �  $  a  U    �  $  2  @  O  T  W  Z  X  S  N  H  ?  7  /  &      �  �  �  H  �  k  �  �  �    H  6  �    +  �  �  ;  ;  Z  ?    k      g  �  �  �      �  �  �  U    �  �  J  �    P  �  q  b  U  J  =  0      �  �  �  l  3  �  �  	  g  �  
  .    (  0  4  6  3  0  ,  $      �  �  �  u  K     �  �  �  I  K  M  O  Q  S  T  V  X  Z  _  g  o  w  ~  �  �  �  �  �  �  �  �  p  T  7    �  �  �  ~  S  ?    �  �  P    �  p    	  	�  
=  
�  
�  
�  
�  
�  
o  
M  
*  	�  	�  	  7    b  H  �  _  �  �  �  �  �  y  Y  3    �  �  B    �  �  �    \  �  U  i  w    �  �  �  �  �  }  q  a  P  ;    �  �  �  Q      $    �  �  {  %  �  �  B    �  �  w  K    �  �  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  X  "  �  �    `  �     �  �  �  �  �  �  �  �  �  �  �  �  p  Z  E  5  $    �  �  �  z  l  \  K  =  /    
  �  �  �  v  J    �  �  �  8  �  �  �  ~  r  f  X  E  1      �  �  �  v  9  �  �     �  P  C  6  )            )  :  J  [  b  Y  O  F  <  2  )  o  �  �  �  �  �  �  �  �  �  q  Q  0    �  �  w  1  �   �    �  �  �      �  �  �  l  )  �  �  X      �  
  $    �  �  �  �  �  p  _  O  ?  .     �   �   �   �   �   z   _   E   *  a  Q  A  1  !      �  �  �  j  C    �  �  �  �  �  �  �  �      �  �  �  �  c  ,  �  �  j  3    �  �  ~  -  �  �  �  �  �    +  0  %    �  �  �  �  T    �  ?  �  �  Q  �  .      �  �  �  �  �    a  A    �  �  �  |  C  �  �  z  5  �  (  �    G  e  w  t  L    �  	  2    �  G  �  	�    /  !  �  �  g  ,  �  �  ;    �  g  �  n  \  �  �  !    ,  l  k  W  I  \  �  �  �  �  �  �  �  p  ;  �  �  1  ^  .   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _    �  (  �  �  �  �  �  �  �  }  o  a  S  F  9  -  #        (  9  J  �  �  �  �  �  ~  r  e  [  Q  D  5  $    �  �  �  �  �  �  �  �  �  �  _  q  x  v  k  [  E  '    �  �  f  +  �  �  �  �         	  �  �  �  w  J    �  �  ~  0  �  t    �  C  0  )  #        �  �  �  �  �  �  s  \  @  #     �   �   �  �  �  }  p  e  ]  V  O  B  -      �  �  �  b  A  "     �  R  2    �    -    �  �  �  m  9     �  m    �  >  �  H  �  �  �    u  g  V  B  ,    �  �  �  �  �  z  F    �  �  	5  	�  	�  	�  	�  	�  	�  	u  	B  	  �  `    �  1  �  /  p  �  `  �  �  �  �  �  �  �  �  �  �  �  �  y  ]  B  �  u    �  |  �  �  �  �  �  w  ]  F  8  *      �  �  �  �  w  K    �  �  �  �  �  �  s  ?    �  �  )    �  �     �  v  8  (  D  3      �  �  �  �  �  �  s  ]  C  '  �  �  �  �     �  �    �  �  �  �    �  �  �  �  �  �  �  �  �  a    �  �  �  �  �  �                �  �  �  �  b      (      �  �  �  �  �  �  u  Q  '  �  �  �  h  &  �  �  �  <  �  �  (      �  �  �    M    �  �  k  '  �  x  �  b  �    `  j  �  �  v  a  L  6    �  �  �  U    �  �  d  !  �      %  �  A  j  �  4  �  �  �    �  �  >  �  j  �  
�  O  �  �  b  H  .    �  �  �  �  z  W  .  �  �  �  k  ;    �  �  O  ]  Z  W  T  Q  N  K  H  E  B  ?  =  <  :  8  6  4  2  1  /  �  �  n  Q  .  
  �  �  �  b  5  	  �  �  �  J  �  u  �  �  �  �  �  �  �  �  �  b  J  2    
  �  �  �  �  �  L  �  �  *  &      �  �  �  �  x  k  \  A    �  �  �  o  7  �    �  �  �  �  {  b  I  0    �  �  �  r  <    �  J  �  �  j    �  �  �  �  h  C    �  �    ?  �  �  s  -  �  �  Y  �  
e  	�  	�  	G  �  �  q  �  o    �  G  �  �    �  u    �  �  D      �  �  �  �  �  n  T  5  	  �  �  5  �  �  S  !  
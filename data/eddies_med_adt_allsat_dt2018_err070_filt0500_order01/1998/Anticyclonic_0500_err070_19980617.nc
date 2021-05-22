CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?����E�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       =�F      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @FW
=p��     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @vpz�G�     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q            t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @��           �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >Z�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��O   max       B1i�      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�   max       B1<[      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�2^   max       C�h�      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�wV      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P���      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�s�g��   max       ?��|���      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       >%      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @FS33333     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vpz�G�     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @���          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E#   max         E#      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�    max       ?��|���        R<      	               [   &   $   6         
      \            &      �               
               B         C   �      1   	               n      )   7         )      
      1               ?O%�vN���N��N�s�NFN���PoGP jO�lSPnU]P
G�ONȚN59N���P���N,��N��N�Q�Pa�NU�P�QO/ʘN���OF��O�3�O<�mO`�~Nc�<O�$SNN><PNyOBd�O��Oւ$P��0Oa%O���NXsN@S N��DNL�LO�6+O���N��O��P^�M���N���PN�2M���N�OK4O���O|��N��N?�]N��OSμ�t���o�u�e`B�#�
��`B�o��o;o;ě�<t�<#�
<#�
<#�
<D��<D��<e`B<e`B<u<u<���<��
<�9X<�9X<ě�<���<���<���<���<�`B<�`B<�h<��<��<��=o=o=\)=\)=�P=�P=�w=�w=#�
=0 �=0 �=0 �=8Q�=D��=T��=aG�=ix�=q��=�C�=��
=��
=��=�F��������������������������������������������������������������������������������sprt{�����wtssssssss(&&&*/2<HPPHEA</((((�#/87@Oaqf[<#
�"/;PZ\TQH;/!���������������������������/<QN<1(
�����������������������CCFHLUanvvz{una^UKHC��������������������#'0<IIEIJIHD<0&$#-*),5BNg�������t[B5-���������������������������������������� �)**)"     $5Ngt�|�|gNB5) �����������������������������
!" 
���#/<AHUY]YUH</.%#�)575/) ��������������������������������	������-06;CO\htqqoih\OMC6-{y�����������������{�������������������������
#%$"
���������������������������)5BNZabaWSK5���������
������������������������������%&���6BKYo�����t[B)���

�����#/<HU\adffaUH<2��������������������/0;<IU\UOI<0////////����#(#������9;BO[]\[OB9999999999OPRWW[amz�����znfaTO��������������������g\gtu����tslggggggg������������������������)18950�����jinuz}~}znjjjjjjjjjj52559BEN[gijg_[NB555������&++"����������������������������ttx������������wvvtt����������������������������
�������������%),,'���������������	|{���������||||||||���������

�������5�B�N�[�`�c�]�[�N�M�F�B�5�1�*�)�+�0�5�5�����������������������軞�����û˻λû������������������������������������������������������������������n�zÇÎÓÖÓÇ�z�r�n�m�n�n�n�n�n�n�n�n�����	��"�%�"����	������������������5�s���������������s�Z�/����������5�H�T�a�m�z�������m�H�;�#����������/�H��������&�!����������Źŵ�����������T�a�m��������������������a�T�B�;�;�J�T���4�A�Z�s�z�~�|�s�f�Z�M�A�(���������#�)�/�(����������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͼ������������������������r�g�_�f�r�s������	�T�a�j�k�`�D��������������~�������#�/�9�<�<�<�0�/�.�#�!��#�#�#�#�#�#�#�#�)�6�=�6�-�/�)����������#�)�)�)�)�A�N�X�Z�c�c�[�Z�N�L�A�5�4�)�5�6�A�A�A�A�.�G�T�`�m�u�������y�m�`�G�;�"�����.���(�.�-�(��������������DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DmDgDhDo����������������������������������M�Z�[�c�e�e�\�Z�V�M�E�A�?�A�C�J�M�M�M�M�r���������������������|�r�Y�E�H�Y�a�r�;�T�a�m�m�m�a�]�K�H�;�/�%�"���"�+�.�;�	��$�.�:�A�A�C�;�.�"��	����������	�����Ⱦʾ;Ͼ˾ʾ������������������������Ŀѿݿ����ݿѿĿ��ĿĿĿĿĿĿĿĿľs��������������������s�f�[�P�M�O�X�f�s�����	�����	��������������������������6�O�V�h�v�y�z�t�h�[�O�B�)���������/�<�G�H�U�_�a�e�a�U�H�<�.�#�����#�/������)�.�1�4�>�6�)���������������_���������ûлٻлû������l�_�C�2�6�Q�_�'�Y�����������f�@�4������ֻڻ�'�����������������y�`�T�L�T�\�`�m�}�������"�:�G�J�@�<�6�,�"��	�����ؾؾ��	�"�ݽ������������ݽڽսݽݽݽݽݽݽݽݺ�����������ںֺ��������⾥�����ʾѾԾ;ʾ��������������������������ʾӾվ̾ʾ����������������������������n�{ŇŠŭŹ������ŸŭŠŔŉł�{�n�l�l�n�Ϲ������	����Ϲù��������������ù�ǈǔǜǗǔǈǇ�{�o�i�n�o�{Ǉǈǈǈǈǈǈ�h�tāčĳĽ����������ĿĦĈā�w�p�d�`�h�#�<�U�e�v�|�{�p�b�I�0����������������#���"�(�-�(�����������������ĽнҽнͽнսֽҽнĽ����������������Z�������v�\�E�5�(������������2�Z��!�-�.�:�E�:�-�(�!�����������F�H�S�[�S�F�:�8�-�!������!�-�:�C�F���������������������y�n�g�\�`�b�l�s�y�����
��#�.�2�0�,��
���������������������5�B�N�[�g�j�x�y�t�p�g�[�N�<�5�1�2�3�3�5ƚƧƳƶƾƳƧƚƚƓƚƚƚƚƚƚƚƚƚƚ���������������������������������������̼'�4�@�M�U�S�M�@�4�'�$�!�'�'�'�'�'�'�'�'EuE�E�E�E�E�E�E�E�E�E�E�E�E�EzEpEiEgEiEu J " b $ U A ` 4 0 9 . E R 8 = V m 0 ; K ( F R R 7 8 V O & g ) 3 D R = e M Z m & A G ) w \ I G f F � s E # < S f ? *  v  �  �  �  Q  �  q  �  d  �  ]  �  ^    �  q  �  �  �  S  o  �    �  E  �    k  @  }  �  �    9  �  �  �  �  �  �  }  "  5  �    �  &    �  >  �  �      D  h  �  ���o��`B��`B�o�ě��o=�E�=t�=�P=u=t�<�1<��
<�1=��<�t�<�1<��=Y�<��
>Z�=8Q�<�=\)=49X=\)=t�<�h=L��<��=��=<j=D��=ȴ9>'�=H�9=��
=0 �='�=D��=0 �=T��>�P=P�`=���=Ƨ�=8Q�=H�9=�E�=]/=��=��
=�/=� �=��=�1=���>:^5B�B��B#NB�-B	�kB�7B7�A��OB�'BlB ��B#�B{�B%�B�cB{}B=Bm{B	zB�B	B�DB/B"�fBNfB1i�B��B3B;�B�oB1B;zB?B�BУBfyB�GB�NB&�B'�B�A�i/B�]B	��B��B�&B�B9|B
kB*s�B�0B,|�B�BY�BrPB��B۟B��B��B�6B#B�B��B	��B�B@#A�BCB?�B ��BB�B%��B	��B��B½B@�B69B�B>�B��B�@B"?�B@:B1<[B
�Bx�BtB�B@B7=B�AB��B��B*B;B��B&�>B�B�`A���BAOB
>dB?mB��B�B�[B�B*A B��B,�AB?�B�rBv2B��B��B�A��8?PwA@�,�A�1A���A�AA�B�A�UA���A��A9�0A�27C�h�@搐A��A��uAՖhA�&�AdA�{C���A҈|A>h�@��A��8A^�BAK�AA{��AC��A���A׾xA�@�A�2=@���@�j�Al�{A[�SA,�U@H`ANAOٶA��4>�2^B��A��A��DA4��A' A�-.@p��@x�&A�&A��%A�!%B�SB�A@��[C��A���?PN@�B%A���AȆ@A��|A�M�A��A�d,A�~A<��A�C�wV@��A��GAA��	A�}rAb��A�"�C���A�v�A>�8@�{�A�w�A\�GAKFA{�KAD=�A� xA؁�AA�o�@�
�@���AnJ�A\�2A-a�@K��AM#^AP�=A�%>��B?�A�Z�A��A4ugA'A��W@k��@��A �A���A��B�-B�@ъ�C��      
   	            \   &   $   7               \            '      �      	                        C         D   �      1   	               n      )   7         *            2               @                     9   %      3   %            =            #      #                              '         %   ?      !                  %      !   +         3            #                                                #            5                                                !            '                              !   +         3                           OKGN���N��N�s�NFN���O��2O��Oh�O�;�O�.PONȚN59N�l>P���N,��NMkN�Q�OwߴNU�O3�N�N���OF��OS@�O$ȽO`�~Nc�<O��NN><O�O�OBd�O[�OOkOP	�Oa%O�0�NXsN@S N��DNL�LO�6+O��N��O��P^�M���N���PN�2M���N�N�x�Oz�O|��N��N?�]N��O-}�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  E  s  /  ^  �  �  �  �  �  ;  �  y  �  �  �  �  �  �  �  �  �  K  �  �  �  ?  |  u  �  F  n  �  ^  /    !  S  �  M  �  p  ʼ�C���o�u�e`B�#�
��`B=C�<u<#�
=C�<D��<#�
<#�
<49X=+<D��<u<e`B<�`B<u=���<���<�9X<�9X<�`B<���<���<���<�`B<�`B=0 �<�h=+=P�`=��w=o=C�=\)=\)=�P=�P=�w=��=#�
=0 �=0 �=0 �=8Q�=D��=T��=aG�=�C�=���=�C�=��
=��
=��>%��������������������������������������������������������������������������������sprt{�����wtssssssss(&&&*/2<HPPHEA</((((	#/<GLSTQKH</#	"/;?DINLH;/"�������������������������
 $#"
������������������������CCFHLUanvvz{una^UKHC��������������������#,0<>IIIGC<80(%#114BN[g�������tgNB51���������������������������������������� �)**)"    '&)4BNY[gnnf[YNB5+)'����������������������������

����#///<HISTHG<:/)$#�)575/) ����������������������������������������6267=CIO\hqoomhh\OC6{y�����������������{��������������������������
## 
��������������������������)5BQY\YPNGB5��������
�����������������������������������)6BUZ\ggYOB6)���

�����#/<HU[`dfeaUH<4��������������������/0;<IU\UOI<0////////����#(#������9;BO[]\[OB9999999999OPRWW[amz�����znfaTO��������������������g\gtu����tslggggggg������������������������)18950�����jinuz}~}znjjjjjjjjjj52559BEN[gijg_[NB555������&++"����������������������������ttx������������wvvtt������������������������������������������%),,'���������������	|{���������||||||||��������	���������5�B�N�[�_�a�[�[�N�K�B�B�5�3�-�+�,�1�5�5�����������������������軞�����û˻λû������������������������������������������������������������������n�zÇÎÓÖÓÇ�z�r�n�m�n�n�n�n�n�n�n�n�����	��"�%�"����	������������������(�5�N�g�q�h�^�N�A�(���� ��������(�H�T�a�h�p�s�q�m�a�H�;�/�"��
��"�/�;�H�����������������������ſ���������m���������������������z�m�a�^�T�R�T�a�m��4�A�Z�s�z�x�s�f�Z�M�A�(�������������#�)�/�(����������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͼ��������������������r�h�f�a�f�r�v���	�"�H�Z�]�X�O�0��	�������������������	�#�/�9�<�<�<�0�/�.�#�!��#�#�#�#�#�#�#�#�)�6�<�6�,�,�)���������$�)�)�)�)�A�N�X�Z�c�c�[�Z�N�L�A�5�4�)�5�6�A�A�A�A�;�G�T�`�f�g�`�^�T�G�;�.�#�"���"�.�4�;���(�.�-�(��������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��������������������������������M�Z�[�c�e�e�\�Z�V�M�E�A�?�A�C�J�M�M�M�M�r���������������������|�r�Y�E�H�Y�a�r�;�H�T�X�`�e�a�]�X�H�;�/�-�#�$�"�'�/�9�;�	��!�"�.�8�;�>�>�A�;�.�"��
�	������	�����Ⱦʾ;Ͼ˾ʾ������������������������Ŀѿݿ����ݿѿĿ��ĿĿĿĿĿĿĿĿľs�����������������s�f�^�S�O�R�Z�[�f�s�����	�����	��������������������������)�6�B�[�m�t�t�h�[�O�B�)�������
���/�<�G�H�U�_�a�e�a�U�H�<�.�#�����#�/������)�,�1�<�6�)����������������뻅���������������������l�Z�S�H�M�S�_�i���'�M�f����������f�Y�M�4�'����������'�����������������y�`�T�L�T�\�`�m�}�������"�4�;�G�>�;�5�+�"��	�����ھپ��	�"�ݽ������������ݽڽսݽݽݽݽݽݽݽݺ�����������ںֺ��������⾥�����ʾѾԾ;ʾ��������������������������ʾӾվ̾ʾ����������������������������n�{ŇŠŭŹ������ŸŭŠŔŉł�{�n�l�l�n�Ϲܹ�������������ܹù������������ù�ǈǔǜǗǔǈǇ�{�o�i�n�o�{Ǉǈǈǈǈǈǈ�h�tāčĳĽ����������ĿĦĈā�w�p�d�`�h�#�<�U�e�v�|�{�p�b�I�0����������������#���"�(�-�(�����������������ĽнҽнͽнսֽҽнĽ����������������Z�������v�\�E�5�(������������2�Z��!�-�.�:�E�:�-�(�!�����������F�H�S�[�S�F�:�8�-�!������!�-�:�C�F�y�����������������y�x�r�l�l�l�v�y�y�y�y���
��"�#�'�%�!���
�������������������5�B�N�[�g�j�x�y�t�p�g�[�N�<�5�1�2�3�3�5ƚƧƳƶƾƳƧƚƚƓƚƚƚƚƚƚƚƚƚƚ���������������������������������������̼'�4�@�M�U�S�M�@�4�'�$�!�'�'�'�'�'�'�'�'E�E�E�E�E�E�E�E�E�E�E�E�E}EuErEkEuEwE�E� J " b $ U A . A , $ 7 E R ( ? V v 0  K ) B R R > 3 V O " g . 3 E < D e K Z m & A G  w \ I G f F � s O  < S f ? %  T  �  �  �  Q  �  �  7  �      �  ^  �  e  q  �  �  �  S  s      �  �  i    k    }  �  �  �  �  �  �  �  �  �  �  }  "    �    �  &    �  >  �  �  �    D  h  �  k  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  E#  �  �  �  �  �  �  �  �  �  �  �  �  z  J    �  �  J      �  �  �  �  �  �  �  �  s  b  N  6      �  �  �  �  �  �  �  �  �  �  �  �  t  e  U  U  X  Y  X  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  e  T  B  0      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  t  m  g  r  �  �  �  u  �  K  �  �    f  �    r  U  $  �  �    �  �  F  �  h  �  �  0  m  �  �  �  �  �  �  �  �  p    �  G  �    �  w  C  p  �  �  �  �  �  �  �  r  W  7  
  �  p  �  R  �  �   �  /  Y  {  �  �  �  �  �  �  �  �  �  �  �  A  �  �  3  �  �  �  �  �  �  �  �  �  �  �  |  {  z  y  v  g  L     �  �   �  �  �  �  �  �  �  �  q  Y  A  ,  4  L  H  @  .    
      �  �  �  �  �  �  �  �  �  |  g  R  <  &    �  �  �  �  B  �  �  �  �  �  �  �  �  �  �  �  �  �  u  _  @    �  �  2  %  [  {  �  �  �  �  ^  -  �  �  �  +  �  '  j  �  �  S  �  �  �  �  �  �  �  �  �  j  S  6    �  �  �  b  ,   �   �   ~  B  C  D  T  �  �  �    5  j  �  �  �  �  �  �  �  �  �  �  s  e  S  A  .      �  �  �  �  S  "     �  �  �  j  A    �      �  �    +  /  (    �  �  �  S    �  W  �  )  �  ^  M  <  +      �  �  �  �  �  �  �  o  Z  E    �  �  �  �  �  �    �  �    T  y  �  �  �  K  �  �  i  �  @  3  
�  �  �  �  �  �  �  �  �  t  T  7      �  �  u  7  �  �  O  �  �  �  p  X  @  &    �  �  �  �  �  w  f  a  a  s  �  �  �  �  |  p  c  U  H  A  >  9  *      �  �  �  
    �  �  M  m  y  �    y  r  g  [  J  6    �  �  �  J  �  �  S  6  3  7  :  6  0  &    
     �  �  �  �  �  �  �  y  P     �  �  �  �  �  �  |  k  Y  F  2      �  �  �  �  �  �  {  X  y  u  q  l  h  d  _  [  V  Q  L  G  B  ;  /  "    
  �  �  �  �  �  �  �  �  }  o  _  M  9    �  �  �  H  �  �  U    �  �  �  �  �  �  �  �  �  �  �  ~  t  h  Y  I  9  )    	  -  `  �  �  �  �  �  �  z  V  #  �  �    �  �  %    %    �  x  Z  ?     �  �  �  �  �  l  Q  5    �  �  a    �    �  �  �  �  �  �  �  �  }  Z  G  8  #    �  �  q  '  �  |  �  g  �  �  �  �  �  �  �  r  8  �  �  :  �  �  N  �  �  �  
�    -  p  �    �  �  �  �  �  p    
�  	�  	8  1  �  �  t  �  �  �  �  �  �  �  �  y  W    �  �  W    �  �  H     �  �  �  �  r  K    �  �  F  �  �  d    �  �    ~  �  �  �  K  <  -      �  �  �  �  �  �  j  R  L  F  ,    �  �  �  �  �  �  �  s  a  N  A  6  *  '  -  2  :  H  U  \  E  .    �  �  �  �  z  i  W  D  -    �  �  �  �  m  C    �  y   �  �  �  �  z  h  V  C  /      �  �  �  �  �  f  G  %     �  ?    �  �  �  �  �       �  �  �  �  �  �  o  <    �   �  
v  ,  �  �  A  l  {  k  H    �  �  B  
�  
S  	�  �  6  {  ]  u  h  P    �  �  �  �  �  �  ~  d  L  5       �  �  �  �  �  �  �  �  �  Y    �  �  $  �  G  �  �  p    �    �  @  F  3    �  �  f     �  �  f  -  �  �  }  '  �  N  �  &    n  i  e  `  \  W  S  N  J  E  A  >  ;  8  5  2  .  +  (  %  �  �  �  �  �  �  �  �  �  |  p  ^  K  8  %     �   �   �   �  ^  M  =  (    �  �  �  �  �  {  ]  5    �  �  2  �  �   �  /  %        �  �  �  �  �  �  �  �  �  �  �  {  m  `  S    %  D  >  7  -  #      �  �  �  �  �  �  �  �  �  �  �  n  �  �  �  �  �  
      !      �  �  �  \    �  �   �  �  �    3  E  N  S  Q  ?    �  �  b    �  2  }  j  g  p  �  k  N  +    �  �  n  6  �  �  �  �  o  ;  
  �  �  �  �  M  I  E  B  >  :  7  3  /  +  '               �  �  �  �  �  �  v  h  U  8    �  �  �  �  �  x  _  P  E  :  0  %  p  _  N  =  +      �  �  �  �  �  �  �  m  X  8    �  �  w  �  �  �  �  �    T    �  t    
�  	�  	R  �  �  �  �  .
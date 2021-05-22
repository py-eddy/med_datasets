CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�5?|�i      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       PBg�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       =�h      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @E�          p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @ve\(�     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @O@           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�F           �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >(��      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�	D   max       B,'�      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��.   max       B+�      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�Y�   max       C���      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C��      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          w      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          3      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          %      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       O��V      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���C,�   max       ?�XbM��      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �#�
   max       =�h      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @E�          p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @ve\(�     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @O@           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�2@          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���@��   max       ?�S&���     �  N�                  "            .      %            !                  
   	                  &   .      v   	   	         b   I      $            /                     %      /   O�2�NBN`�N��N5�=OASO��]OM
O+�MO��
NQ��Oq�O��QOɘyN���N�~�OI�eO*N>nN�$NM�N��NU�%OX�N�
cNd�Oc�Ob}PBg�O���N$(�P8��N�[iN�`VN��:N���O�6�O�Q6N�mxO��O��N/GN��O��SO�p�O�fNͅ~N�dM���O/O:��OJ�xO��RO����t��#�
�ě��o��o;�o;��
;ě�<#�
<e`B<�o<�C�<�t�<���<�1<�9X<���<���<�/<�`B=o=+=C�=C�=\)=t�=��=#�
=0 �=8Q�=8Q�=@�=@�=@�=P�`=T��=T��=Y�=Y�=]/=e`B=e`B=q��=y�#=y�#=��=��P=��P=��-=�9X=�9X=\=�"�=�h	)5B[gvz�tgOB)[Z[goty{tjg[[[[[[[[[=9:?BOOUTONB========#'010,#016BJORVOHB600000000f`]]agty���������tgf����
0IbifS0����""#/<HUW__ba_UH</"������������������������������������������������������������ ��	"/9;@=/%"	��������������������3+,6BGO[hw~��}{th[M3').5:6695) ���������������������� 
#/17:></'#
�������������������������������������������������������������������������������������������������� �����������/<BIGMGDD<:4--)$�yux������������������!���������������""
���������������bc_dt�����������tnlb��������������������#0:0.#������,353)�����������������������������#)2-)������^\`gkt��������tg^^^^����������������������������
 " �������)-.-11/)���.156BEN[\b`[PNB5....)6BO[dfOB?6)�������������������������


�����������������	����������SVTVZjmz���������zaS��������������������$ #&).5BCGJIHFB5*)$$A@BFN[git~thg[WNHBAA|���������������||||"&'"����������������������������������������]WUV\_anz������}zna]������

�������#
��������
#&)$#�G�T�h�w���������y�`�T�B�>�.�(�&�*�.�=�G�����������������������������������������l�y�������������y�m�l�g�l�l�l�l�l�l�l�l��������޼ּ˼ּ׼��������ʼּۼ����ּѼʼļżʼʼʼʼʼʼʼ��n�zÇÓàåâàÝÕÓÊÇ�z�n�j�f�d�l�n����������������������r�M�M�Y�f�r�{�y��������������������������������뼱���ʼҼּڼּ˼���������������������������������ܻлû������������������ûܻ��M�Z�\�^�Z�M�A�4�+�4�A�B�M�M�M�M�M�M�M�M�;�H�T�a�q�z�}�����z�x�m�a�H�;�6�/�-�5�;�ʾ׾��	�"�.�4�(��	�����׾ʾ¾����þʾs������������Ⱦɾ��������s�f�Z�Q�R�Z�s����������������������������������������àìù����������ùðìàÜÓÒÓ×Ýàà�m�v�y�~�y�p�c�`�T�G�;�,�*�.�/�;�G�I�`�m�B�O�[�d�h�t�v�y�t�h�[�O�B�6�5�6�6�=�B�B���*�+�-�*���
������������/�<�G�H�U�a�a�b�a�U�H�<�/�#���#�$�/�/�a�m�r�z�z�z�w�m�g�a�U�[�a�a�a�a�a�a�a�a���ùùϹܹ߹����ܹϹ˹ù������������������������������������������������������	���	�������������������������������/�<�H�U�X�Y�U�M�H�=�<�/�(�#�"�#�#�-�/�/��'�3�9�5�3�1�'�"��������������)�,�5�E�H�:�5�)�������������
���"�.�;�G�T�a�_�T�R�G�;�.�"��	����������"�+�-�*�&��	�����������n�p�������5�A�N�Z�^�e�k�s�s�g�N�A�5�(�!����(�5�ֺ�����غֺкӺֺֺֺֺֺֺֺֺֺ��������,�1�-��������ÿàÊÇÓáù���H�U�a�n�q�w�n�a�U�H�D�=�H�H�H�H�H�H�H�H�f�i�j�l�h�f�c�Z�M�F�E�H�M�R�Z�c�f�f�f�f���(�3�5�8�8�5�,�(� ���	���������������������������v�v����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��#�0�<�A�G�G�E�<�0�#��������Ĺĺ�������#�I�U�a�b�m�n�q�o�n�b�U�O�I�H�F�E�I�I�I�I�e�r�����������Ǻƺ����������~�o�d�\�^�e���������������������������������������ž��ʾ׾���׾Ͼʾ����������������������нݽ�����������ݽԽнͽ˽ннннн��{ŇŔŠŭ��������ŹŭŠŔŇ�{�v�s�s�y�{�����������½ʽŽĽ������������u�s�r�y��Ƴ����������������������ƳƧƜƜƧƨƳƳ�O�[�b�h�m�n�h�h�_�[�Z�O�N�F�B�<�B�K�O�O�h�i�o�t�u�t�p�h�[�O�K�I�M�O�[�]�h�h�h�h�#�(�#���
��
��#�#�#�#�#�#�#�#�#�#�#�ܻ�����'�4�7�4�*���������ܻ׻׻ܼ@�Y�f�r�������������������x�f�Y�E�9�@�����������ɺκҺ˺ĺ�������������������E\EuE�E�E�E�E�E�E�E�E�E|EsEsEoEiEZEVEWE\�a�U�S�U�W�a�b�n�p�{ŁŇŋōŎŊŇ�{�n�a F ] 0 % a B e 9 0 X L 9 5 ) E ( V 2 3 L 0 n : } D J $ & [ 9 Z > V l $ ; & X L c M w @ ! ; B L E I Z y  ] .  X  [  s  1  {  �  �  �  |  v  e  �  ~  �  �    �  +  Z  &  d  �  [  G  �  �  �  �  �  5  c  c  �  ,    �  �  P    �  R  _  �  k  U  T    �    �  �  �  ]  L<t���o;��
;o;��
=��=+=+=o=u<�j=Y�=�P=8Q�<�`B=]/=0 �=L��<�h=#�
=,1=0 �=,1=8Q�=Y�=#�
=m�h=�o=��T=�Q�=T��>(��=aG�=e`B=q��=m�h>�u>   =�hs=�E�=��=y�#=�%=�"�=� �=�{=�E�=�v�=��
=�
==��=���>�->t�B�AB	\BfHB%9B+�B
!�B%`gB��B W�B"�B>"A� B 1eBM)B�B!ݴBx�B�B\rB��B1�B �B+-B�KB`�BIBXoBm B
�mB�(B%BB��B�B��B	�TBb�B3�B�B�B�.BiBZ@B#T�A��B,'�B[EB�B�A�	DB�B�B�B�@B�hBC�B	?�BA�B%?�BI�B
/�B%�LB�BB F�B"�!BE2A��.B ?B@'B��B"=�B>�B�1BWyB�B;B ?�B7B_BBB@BAAB��B
��B��B%;�B��B�UB��B	��B?�B?�B3�B>CB��B@�B?�B#}�B 8jB+�BCXBAQB;�A�;BToBH�B5�B�B;�Af�kA���A��Ac�A �	A�<�@��IAҘ?@�g%@�]A<�A�0AW|nAG0A�:�A̿Ag?A��A�O@A�LA��>�Y�AI;�A�M4A�y�?��XA��Aa�*A�.�A��@Ab�A�O�A��+A>��A���A��MC���A田A�s@��A��AR)A+�'A�ZA ��B�AA�1<A�ԎA�0@�b@�@�?C���A��Ae.�A�JfA2A��A ��A�	@��A��@��@���A<5wA�pwAU*#AH�=A�|LÄ�Ah��A�q�A�~ A8A�bD>���AI�A�ZwA��?��A�Q�AaA�%A��x@C��A�CA�b�A? EA���A��C��wA�؇A�A*@�A���AQ�A,�sA�uA�B0�A�*<AډlA�@��@�� @��C��A�UQ                  #            /      %            !                     	                  '   .      w   	   
   	      b   I      $            /                     %      0      #                  -                                                   !               3         1                  !                                 
                                    %                                                   !               !         !                                                   
               OPJ�NBN`�N��N5�=OASO��VNٓ'N?�pOT�NQ��N�%>Oi?�O["�N���N�p�O8#�O*N>nN���NM�N��NU�%OX�N���Nd�OSG�Ob}O�2�OZz�N$(�O�3�N�[iN���N��:N���O���O�K#N�mxO��O��N/GN��Om�JO�p�O�fN�	�N�dM���O/O�OJ�xO��RO��  �  �    �  b  �  6  �  1  �  t  ,  ;  �  �  6  /  �  %  B  �  $  �  A  $  �  T  �    �  h  !    G  �    x  �  �  �  �  �    �  �  �  M  (  E  m  �  ,  
  ���`B�#�
�ě��o��o;�o<o<e`B<�9X<�h<�o=+<�j<�`B<�1<�<���<���<�/<�=o=+=C�=C�=�P=t�=�w=#�
=q��=T��=8Q�=� �=@�=H�9=P�`=T��=�+=�O�=Y�=]/=ix�=e`B=q��=�O�=y�#=��=���=��P=��-=�9X=��=\=�"�=�h $)05BN[a[[XWNB5) [Z[goty{tjg[[[[[[[[[=9:?BOOUTONB========#'010,#016BJORVOHB600000000f`]]agty���������tgf��#0IdebPE<0#��)(%,/<?HMUVURH@<9/))������������������������������������������������������������	"/34/-#"	��������������������>=BHNO[hotuwwtmh[OF>').5:6695) ����������������������
#/06=<4/,#
��������������������������������������������������������������������������������������������������� �����������/<BIGMGDD<:4--)${x������������{{{{{{��!�������������� 
����������������qqssu������������xq��������������������#0:0.#�������! ������������������������� ).*)^\`gkt��������tg^^^^����������������������������

�������� $'*--+'.156BEN[\b`[PNB5....)6BO[dfOB?6)�������������������������


�����������������	����������[]bmz����������zqmb[��������������������$ #&).5BCGJIHFB5*)$$A@BGN[fgtgg[UNIBAAAA|���������������||||"&'"����������������������������������������]WUV\_anz������}zna]������

�������#
��������
#&)$#�;�G�T�`�f�m�q�v�v�s�m�`�`�T�G�;�1�0�2�;�����������������������������������������l�y�������������y�m�l�g�l�l�l�l�l�l�l�l��������޼ּ˼ּ׼��������ʼּۼ����ּѼʼļżʼʼʼʼʼʼʼ��n�zÇÓàåâàÝÕÓÊÇ�z�n�j�f�d�l�n�������������������r�f�^�P�N�Q�Y�f�r�������������
����������������������뼱�����ü¼������������������������������ûлѻܻݻܻԻǻû��������������������þM�Z�\�^�Z�M�A�4�+�4�A�B�M�M�M�M�M�M�M�M�T�a�b�m�o�t�q�m�a�W�T�H�G�C�H�P�T�T�T�T�׾����"�%�#���	���׾ѾɾžǾʾѾ׾������������������������r�f�`�c�j�s�����������������������������������������ìùþ��������ùìà×Üàåìììììì�m�s�y�}�y�o�b�`�T�;�5�.�-�.�1�;�G�K�`�m�B�O�[�d�h�t�v�y�t�h�[�O�B�6�5�6�6�=�B�B���*�+�-�*���
������������/�<�C�H�L�Q�H�<�/�#���#�,�/�/�/�/�/�/�a�m�r�z�z�z�w�m�g�a�U�[�a�a�a�a�a�a�a�a���ùùϹܹ߹����ܹϹ˹ù������������������������������������������������������	���	�������������������������������<�H�S�T�K�H�<�<�/�,�$�%�/�1�<�<�<�<�<�<��'�3�9�5�3�1�'�"��������������)�5�C�G�8�5�)������������������"�.�;�G�T�a�_�T�R�G�;�.�"��	�������������	�����	���������������������(�5�A�N�Z�Z�a�d�e�d�Z�N�A�5�(�%���"�(�ֺ�����غֺкӺֺֺֺֺֺֺֺֺֺ������������ �������������øòõ�����H�U�a�n�q�w�n�a�U�H�D�=�H�H�H�H�H�H�H�H�Z�f�g�j�f�f�_�Z�M�K�H�L�M�W�Z�Z�Z�Z�Z�Z���(�3�5�8�8�5�,�(� ���	���������������������������v�v����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D���������#�0�4�;�:�0�#������������������I�U�a�b�m�n�q�o�n�b�U�O�I�H�F�E�I�I�I�I�e�r�����������Ǻƺ����������~�o�d�\�^�e�������������������������������������������ʾ׾���׾Ͼʾ����������������������нݽ�����������ݽԽнͽ˽ннннн�ŔŠŭŽ��ŽŴŭŢŠŔŇŇ�}�x�x�{�}ŇŔ�����������½ʽŽĽ������������u�s�r�y��Ƴ����������������������ƳƧƜƜƧƨƳƳ�O�[�a�h�l�m�h�f�[�P�O�G�B�=�B�L�O�O�O�O�h�i�o�t�u�t�p�h�[�O�K�I�M�O�[�]�h�h�h�h�#�(�#���
��
��#�#�#�#�#�#�#�#�#�#�#�ܻ�����'�4�7�4�*���������ܻ׻׻ܼf�r�����������������������r�f�Y�Y�\�f�����������ɺκҺ˺ĺ�������������������E\EuE�E�E�E�E�E�E�E�E�E|EsEsEoEiEZEVEWE\�a�U�S�U�W�a�b�n�p�{ŁŇŋōŎŊŇ�{�n�a 4 ] 0 % a B ` ' , / L 1 2 $ E * Z 2 3 - 0 n : } B J " & C + Z - V X $ ; ' [ L c I w @  ; B 8 E I Z `  ] .  �  [  s  1  {  �  1  �  Z  g  e  �  �  �  �  �  �  +  Z  �  d  �  [  G  �  �  �  �  �  �  c  �  �  �    �    �    �  8  _  �  �  U  T  �  �    �  B  �  ]  L  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �  �  �  �  �  �  �  �  �  �  �  �  v  I  �  y  �  �    �  �  �  �  �  �  �  �  �  �  �  �  t  g  [  O  O  �  �  �    
  �  �  �  �  �  �  �  h  H    �  �  d    �  |  )   �  �  �  �  �  �  �  �  �  �  �  |  p  c  W  J  3     �   �   �  b  d  f  i  l  p  s  v  y  |  ~    �  �  �  �  �  �  �  �  �  �  �  �  f  A    �  �  }  L    �  �    �  2  �  7  [      5  5  0      �  �    
    �  �  �  �  �  8  �   �  �  G  q  �  �  �  �  �  �  �  �  �  a  .  �  �  1  �  :  �  v  �  �  �  �  �      '  .  1  ,     
  �  �  )  �  "  �  �     U  �  �  �  �  �  �  Z  !  �  �  +  �  '  x  �  .  U  t  l  d  [  R  I  @  6  -      �  �  �  �  �  l  I  &    Y  �  �  �  �      %  +  +      �  �  Z    �  M  �  g  �    (  6  :  ;  5  )    �  �  �  �  `  0  �  �  @     �  c  k  q  y    �  �  �  �  q  Z  ?    �  �  �  :  �  d   y  �  �  �  �  �  y  h  V  C  1               '  ;  T  l  �    !  +  4  5  1       �  �  �  S    �  z  1  �  �  6  %  /  .  #    
  �  �  �  �  �  �  �  t  F    �  u  '  �  �  �  �  �  �  �  �  n  Q  1  F  t  r  U  2  
  �  h  �  N  %              �  �  �  �  �  �  �  �  �  �  �  {  n  .  *  /  ?  =  7  /  #      �  �  �  �  �  �  �  �  i  F  �  �  �  o  S  5    �  �  �  �  �  ^  6    �  �  �  �  `  $    �    �  �  �  �  p  =    �  �  �  N    �  �  M    �  �  �  m  V  A  -    �  �  �  �  �  k  G  -    �  �  �  A  /           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    #  "        �  �  �  �  �  Z  ,  �  �  T  �  )  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  >     �  O  T  P  B  *    �  �  �  �  |  W  )  �  �  h  
  �  <   �  �  �  �  �  �  �  �  }  b  @    �  �  �  J    �  �  E          �  �  �  �        �  �  �  �  a    �  �  ?  @  }  �  �  �  �  �  �  �  �  q  <  �  �  %  �  F  �    ?  <  h  b  [  S  H  >  2  &        �  �  �  �  �  �  �  �  �  �    �  �  �         �  �  s    �  +  
�  	�  �  �  �      �  �  �  �  �  �  r  X  =  !    �  �  �  H  �  y    �  +  1  7  >  E  E  C  ;  /  !       �  �  �  �  T  $  �  �  �  �  �  �  �  �  u  d  R  A  .      �  �  �  ~  Z  5      �  �  �  �  �  �  �  u  \  =    �  �  �  }  P     �   �  �  K  n  u  b  *  �  q    �  �  o  �  j  
�  
  �  m  �  �  2  m  �  �  �  �  �  �  \  $  �  J  
�  
*  	�  �  L  �  e  �  �  �  �  r  X  6    �  �  �  l  8    �  �  \  =  -    �  �  �  �  y  \  ?      5  $  �  �  �  :  �  U  �  V  �  �  �  �  �  o  ]  K  8  #    �  �  �  �  a  -  �  �  ?  �    �  �  �  �  �  z  \  ?        �  �  �  �  e  F  )     �   �         �  �  �  �  �  �  �  �  �  �  w  h  R  9        �  8  m  �  �  �  �  y  l  ^  I  /  	  �  �  P  �  �    r  �  �  �  �  �  �  �  �  l  =    �  �  W    �  |  .  �  �   �  �  �  �  �  v  a  H  ,    �  �  �  �  u  N    �  �  9   �  .  F  8    �  �  �  �  t  E    �  �  W  �  �  9  �  `  �  (      �  �  �  r  F    �  �  p  5  �  �  X  �  �  %  �  E  9  -  !    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  _  C  #    �  �  �  k  :  	  �  �  �  �  t  6  �  �  �  8  V  n  �  �  �  �  �  ~  I    �  �  U  �  �     �    �  ,    �  �  �  �  �  �  o  S  2    �  �  p  "  �  D  �  y  
  	�  	�  	�  	o  	_  	~  	�  	u  	Z  	  �  K  �  ]  �  M  �  A    �  u  =  $  �  �  }  @  �  �  f    �  M  �  �    �    �
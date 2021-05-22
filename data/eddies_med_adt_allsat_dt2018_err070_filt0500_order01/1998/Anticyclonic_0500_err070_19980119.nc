CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�vȴ9X      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �]/   max       =�F      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @E��z�H     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @vM�����     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @��@          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �,1   max       >F��      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��T   max       B,�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|�   max       B,�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�F�      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?19�   max       C�OZ      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          w      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P.?�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��O�;dZ   max       ?ԋC��%      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �T��   max       >n�      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @E��\(��     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�|    max       @vM�����     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?x�t�j~�   max       ?ԋC��%     �  T            	                     Z   	         K                  S   	               Y   I         6         !   w         @      \                  	      '   -   !   w      3   (      	      	      KN��Nw8�Nyi�N��O�?Nn�dN���OQ(?O]CN��P���Ou��N�{HN>׋O�#�O|�ON�Q�N��N�˪N`WP@F:NoN[NG�N=�wNxo4P	jP�.NZ&lO���O!pN>�N-��O;�P[��O�RKOG<�POL�O��PAq�O=�N^5wO��N��N��;N�6Ny]�O�iDOO\vO���Pc�N�q�P��O���O,��NddO:M;NQ-O��O��ѽ]/��P��t���C��D���D��%   ;D��;ě�<t�<#�
<#�
<#�
<#�
<e`B<�C�<�t�<���<��
<��
<�1<�9X<�j<ě�<ě�<ě�<�/<�`B<�h<�h<�<�<��=+=+=C�=�w=#�
=,1=0 �=0 �=0 �=@�=D��=D��=L��=L��=T��=T��=aG�=aG�=ix�=u=}�=�+=�+=�\)=��P=�v�=�Fpprt{�����������}tpp}����������� #-08<=<940#GDNN[fgjpjg[XNGGGGGG��������������������qtu�������ytqqqqqqqq>>DHTU`anpnjjbaWUH>>��������������������srtz��������������ts�����������������������&������!")7>BNPUTMMHB5$ #(/7<HJPTPH<;/+#  ����������������������������������������	"0<IKLTNJH<0#	)5:=575)'����������������������������������������gghrt����~thggggggggGFHUanz���������naQG��������������������98<HTUUUH<9999999999�������������������������������������������������������)#"",6BO[^nqrsrm[B6)�����
/?NUWUK</#	������������������������������

������������

��������
 
���������3,16=BDGOPOBA6333333������������������������5JNglmXB/��"/;@B@;3$"��������������������fftn\`�����������tjf������� 
 #)*#
������/KRQPIB5)��98:>BHUanynkacaVUH<9`]bjn{{�{nkeb``````�������

�������^[][[[agmqzz}}zwma^^��������������������)*5696/)79;BBMO[\][OGB777777���
<>9=3/,#
�����������������������������! ����xqpt����� �����x��������� ���������������)5??5)�����"/;HTaedc[TOH;/"T[amz~���}zqma`YUVTT��������������������������������������"!"##./0010/-#"""""#/3<@GHD<2/-#����������� ���������a�n�zÀÇÍÐÇ�z�q�n�k�a�[�U�T�U�X�a�a�[�g�t�w�t�r�g�c�[�P�X�[�[�[�[�[�[�����������������������x�u�t�x�����������n�{ŇŇŏŇŅ�{�n�b�`�`�b�j�n�n�n�n�n�n�ܹ�����������ܹϹù����ùϹ۹ܺY�d�e�p�p�e�Y�L�L�J�L�P�Y�Y�Y�Y�Y�Y�Y�Y�����������������Ⱦ����������������������*�6�9�C�L�N�N�E�C�6�$��������)�*�ѿݿ����� ����ݿٿѿĿ��������ɿпѹ��������������������������������������������������s�N�(���	�5�H�Z�s����������������'�$���������ſž����D�EEEEEEE	ED�D�D�D�D�D�D�D�D�D�D��ûͻ̻λĻû������������ûûûûûûû���������������
���������������������������������������������r�b�V�S�Y�_�r��Z�f�n�n�o�m�f�Z�M�A�;�>�A�F�M�N�Z�Z�Z�Z�������������	���������/�6�;�H�T�_�a�b�a�T�H�>�;�/�&�"��"�(�/�������ûŻϻû���������������������������(�A�f�l�l�q�������g�N�(����������)�3�2�3�)�����$�)�)�)�)�)�)�)�)�)�)�������������������������������������������#�(�%����
��������������������������������������������������������������������������������������������ܻ����4�L�^�g�c�M�4�'�����ܻ̻ʻͻܿ;�G�T�`�y���������y�m�G�;�.�"���.�9�;������������������~�������������������������������������������s�c�Y�[�f�s����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{D�D�D�D�D��a�h�n�t�z�~�z�n�k�a�_�^�a�a�a�a�a�a�a�a�����ʼ̼ʼļ�������������������������������������������������������øîîù������������)�7�?�J�[�c�P�B�6������öòú�����"�H�T�a�m�s�w�s�m�a�T�=�/�	������������'�.�1�'���������ݹ�������������	��"�'������������^�L�F�P�e�����������ʾ׾���׾ʾ������s�f�[�`�g�p��������!�����������ƚ�w�i�n�zƊƠƳ��ÇÓàìùþùðòìãàÓÇÅ�z�y�zÃÇ�4�A�H�A�>�5�4�(�����(�0�4�4�4�4�4�4��������������������������ŹųŸŹ�������h�tāčĚĦĬĦĠĚĎčā�t�j�h�a�a�h�h����������������������ﺋ���������������������������������������������ɺʺɺɺ����������������������������B�L�N�[�s�g�[�N�)� �������������Óàìù����������ùìàÓÐÇÂÁÇÉÓ�#�0�;�I�P�Q�J�<�0�#��
�������������
�#��-�_�x���v�l�Q�M�F�:�-���ݺ��������������������������������������������ſ������������������y�m�Z�T�W�^�k�{������������&�3�<�C�H�F�<�#��
��������������ŔŠŧŭŨŠŔŇ�{�n�g�b�^�_�b�n�{ŇœŔ�h�tāĂČā�t�l�h�d�h�h�h�h�h�h�h�h�h�h�y���������������������������y�q�r�o�m�y���	��"�/�/�/�"���	�������������������b�o�{ǈǔǕǗǔǎǈ�{�o�b�Z�V�U�V�[�b�b²¿���������������¿²¦¦² & 4 G 5 d , n + ; G c Y J h ) : I a M V < E 4 B E   =  2   . � s 3 [  4 o d K < ` ) X 8 & D f ! 1 a ( i A 9 J A } $ I  �  �  �  �  w  ~    �  g  5  �  (    �  �    4  f  #  ~  F  A  :  p  c  o  ~  ?  h  H  Z  �  g  �      �  =  �  �  a  t  Q  1  �  �  �  j  �      �  �  �  w  9  �  �  "  ��,1���T���t�;ě�<t�<o<���<�o<e`B=���<���<�9X<T��=�9X=��<�<�9X=o<�=��<��<�<�h<�<��=�h=��=C�=e`B=���=C�=�w=��>��=y�#=]/=��=�O�>
=q=aG�=L��=��=y�#=T��=q��=aG�=�Q�=ě�=�->1&�=�\)=�S�=��=�1=���=\=��=�x�>F��B
U�B
��B%n�B��B�GB�zB̣B{�B
�9BpIB�qB��B�1B!`B�B%�0BjB!w�Bx&BZ�B��BAEB/\B��B:�B�cB��B�B��B�eB2B�!B1�B��B9^A��TB!�B
�B$'kBb@B�*B(J�B�tA�}B,�BB~�B	�B!ƚBB�BP�B'B�A�ٳA�X�B��B+�B�BI�ByB
@eB
�tB%D�B�zB��B�B=B��B
�kBC�B�XB�WBT�B!�0B��B%��B@qB!��B?BBMQB>JB@�B5�B�YB?yB�lB�zB?�B�lBB>�B�BB�B<*A�|�B!!�B
��B#�B@>B��B(CFB�A��#B,�B;�BNB��B";�BB`B7zB�YB��A���A��WB�sB+�pB��B=�B?lA�yA���@��[A��?��?��AKL�A���A|qH?:^%A�#<A��C�F�@�dmA�^�@�E�A?ky@��]A���@���A���A�<eA��BA��AK9WA� @�؋Ah�AG�uAF1C���AǾ@���Aϊ�A���A�*?Y��A���AJ;�B&�A�?�A8_A���A���@Wp�@��@%�HA���A�5A�@yZwA�w�Ao�A�y�A�LAܪ1A �A���BSA���A�u,A�|{@���A���?19�?ؼ�AKNA�~CA}��?O^�A��uA�%�C�OZ@���A��@�ueA? @�'8A�~�@��`A�{�AՓ�A�J�A�q�AK>LA��U@�6"Ah��AG*hAE��C���A�}@�@A�T1Aҏ�A���?QHeA�m�AJ��B3�A��A7H�A���A�~�@V"&@�@%��A��AA�q0A���@{vA��]Ao&�A��A���A�~A�	A��B�A�}            	         	            [   
         K                  S   
               Z   J         6      	   !   w         @      ]                  
      '   -   !   w      4   (      
      	      L                                 ;                              -                  '   %                     3         7   !   -                        +         3      )                                                                                    )                                                   /      !                        %               )                     N�FLN</9Nyi�N��N��N7UzN���O@�O]CN��O�3EOu��N�{HN>׋O���O|�ON��.N��N�˪N`WPtJNoN[NG�N=�wNxo4O�WO�jfNZ&lO]��N�ZN>�N-��O*�nO�
KO\7	OG<�P.?�OU\O�j0O=�N^5wN�dsN��N��;N�6Ny]�OͲ�O($ONx�O�N:3�P��O���O,��NddOv�NQ-O��O(��  �  @  O  �  �  9  ;  �       	    �  �  
�  �  A  m  �  s  	  �    �  B  �  
�  	�    ]  #  M  P  g  �  �  �  	#  �  
�    �    �  �  �  q  I  �  y  y    z  �  }  �  -  )  �  z�T���t���t���C��#�
��o%   ;�`B;ě�<t�=e`B<#�
<#�
<#�
<��<�C�<�1<���<��
<��
=+<�9X<�j<ě�<ě�<ě�=y�#=]/<�h=\)=�w<�<��=\)=���=��=�w=@�=@�=�O�=0 �=0 �=H�9=D��=D��=L��=L��=]/=m�h=q��=�l�=}�=u=��=�+=�+=���=��P=�v�>n�qqtt�������tqqqqqqqq�~������������������ #-08<=<940#GDNN[fgjpjg[XNGGGGGG��������������������stx�������~tssssssss>>DHTU`anpnjjbaWUH>>��������������������srtz��������������ts����������������������������������!")7>BNPUTMMHB5$ #(/7<HJPTPH<;/+#  ����������������������������������������	"0<IKLTNJH<0#	
)56950)





����������������������������������������gghrt����~thggggggggLKOUanz��������znaUL��������������������98<HTUUUH<9999999999�������������������������������������������������������1.-.16BOS[aeeda[OB61	#/<@HLMKD</#	���������������������������

�����������


����������
 
���������3,16=BDGOPOBA6333333������������������������)6AB?:5#��"/;=@>;71+"��������������������niimxu������������tn������
  
�������);CGEB95)
�98:>BHUanynkacaVUH<9`]bjn{{�{nkeb``````��������		���������^[][[[agmqzz}}zwma^^��������������������)*5696/)79;BBMO[\][OGB777777����
!/6;2-*"
������������������������������������������������������������������������������)5??5)�����"/;HTaccaZTMH/"T[amz~���}zqma`YUVTT��������������������������������������"!"##./0010/-#"""""#/3<@GHD<2/-#���������������������a�n�z�{ÆÄ�z�n�a�]�V�[�a�a�a�a�a�a�a�a�[�g�t�u�}�t�g�g�g�[�S�Z�[�[�[�[�[�[�[�[�����������������������x�u�t�x�����������n�{ŇŇŏŇŅ�{�n�b�`�`�b�j�n�n�n�n�n�n���������������ܹϹŹŹϹܹ߹��Y�_�e�n�k�e�Y�P�L�K�L�S�Y�Y�Y�Y�Y�Y�Y�Y�����������������Ⱦ�����������������������*�6�B�C�G�H�C�<�6�*����������ѿݿ����� ����ݿٿѿĿ��������ɿпѹ��������������������������������������������������������g�Z�Y�X�[�g�s����������������'�$���������ſž����D�EEEEEEE	ED�D�D�D�D�D�D�D�D�D�D��ûͻ̻λĻû������������ûûûûûûû��������������������������������������˼����������������������r�b�V�S�Y�_�r��Z�f�i�k�k�g�f�Z�T�M�E�J�M�T�Z�Z�Z�Z�Z�Z�������������	���������/�6�;�H�T�_�a�b�a�T�H�>�;�/�&�"��"�(�/�������ûŻϻû���������������������������(�A�O�Z�c�h�����s�N�A�(������������)�3�2�3�)�����$�)�)�)�)�)�)�)�)�)�)�������������������������������������������#�(�%����
�������������������������������������������������������������������������������������������������'�5�B�D�@�;�4�'������޻ܻ���T�`�m�y�������}�y�j�`�T�G�;�2�-�4�?�G�T������������������~�������������������������������������������s�j�f�^�a�f�s��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��a�h�n�t�z�~�z�n�k�a�_�^�a�a�a�a�a�a�a�a�����ʼ̼ʼļ�����������������������������������������������������úùïïù�������������"�+�-�)���������������������"�/�H�T�a�m�q�t�o�m�a�T�E�;�/�"�������'�.�1�'���������ݹ�������������������������������d�S�N�V�n���������ʾϾ׾ξ���������s�h�m�s�w����������������������ƳƠƎƈƇƎƝƧƳ��ÇÓàìùþùðòìãàÓÇÅ�z�y�zÃÇ�4�A�H�A�>�5�4�(�����(�0�4�4�4�4�4�4����������������������ſŹŵŹź���������h�tāčĚĦĬĦĠĚĎčā�t�j�h�a�a�h�h����������������������ﺋ���������������������������������������������ɺʺɺɺ����������������������������)�@�J�N�[�d�[�N�)��������������Óàìù����������ùìà×ÓÇÅÄÇÈÓ�#�0�<�I�M�O�I�F�<�0�#��
��������
��#�:�F�S�_�^�^�U�S�F�:�-�!������!�-�:���������������������������������������ſ������������������y�m�Z�T�W�^�k�{�����������
��#�1�:�@�E�@�<�0��
������������ŔŠŧŭŨŠŔŇ�{�n�g�b�^�_�b�n�{ŇœŔ�h�tāĂČā�t�l�h�d�h�h�h�h�h�h�h�h�h�h�y�������������������������}�y�t�t�t�w�y���	��"�/�/�/�"���	�������������������b�o�{ǈǔǕǗǔǎǈ�{�o�b�Z�V�U�V�[�b�b²¿������������������¿²¥¦¨² # 0 G 5 ` 1 n  ; G ( Y J h  : E a M V G E 4 B E   0  2 $ & � s ) 8 b 4 h Z ? < `  X 8 & D `  + ? 6 i @ 9 J D } $ -  �  N  �  �    R    ,  g  5  _  (    �  �    �  f  #  ~  �  A  :  p  c  o  �  
  h  �    �  g  j  l    �  �  �  "  a  t    1  �  �  �  	  b  �  $  Z  �  ^  w  9  G  �  "  s  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  U  l  |  �  �  �  |  t  j  [  F  )     �  �  t  A    �  o  <  =  ?  @  =  9  5  -  #      �  �  �  �  �  �  �  �  z  O  C  7  *      �  �  �  �  �  �  ~  e  L  3   �   �   |   <  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  f  W  G  8  (  �  �  �  �  �  �  �  �  ^    �  �  H  �  �  p  0  �  �  �  ,  2  6  9  4  +      �  �  �  �  �  Q    �  �  O    �  ;  7  3    �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  V  ,  �  �  ~  %  �  �     �  �  �  �  �  �  �  �    b  B  #    �  �  �  p  W  ?            �  �  �  �  �  �  �  o  R  5     �   �   �   �  [  �  �  �  �    5  �  �  	  	  �  �  9  �  0  W  <    �      �  �  �  �  �  v  \  C  *    !  !          �  �  �  �  �  �  �  q  [  D  +  
  �  �  �  ^  #  �  �  x  @    �  �  y  r  j  b  Z  R  J  B  :  2  *  $  $  $  $  $  $  $  	�  
  
N  
n  
  
~  
p  
Y  
>  
  	�  	�  	>  �  F  �  �  �  �  �  �  �  �  �  w  r  u  �  �  �  q  Y  >    �  �  �  4  �  F  �  !  7  <  ?  @  ?  ;  4  '    �  �  c    �  r     �   k  m  h  c  ^  Y  T  O  F  :  /  #      �  �  �  �  �  �  �  �  �  �  �  �  |  f  M  3    �  �  �  �  W  *  �  a  �  �  s  q  o  m  k  g  c  ]  V  N  B  2  "       �  �  ;  �  U  �  �  		  	  	  �  �  �  s  2  �  w    �    �  P  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  �  L    �  �  w    
                          	  �  �  �  �  �  �  �  �  �  �  �  �  w  k  _  R  F  :  /  #      "  )  0  B  1        �  �  �  �  �  �  }  d  H  &    �  �  �  |  �  �  {  r  f  Z  L  >  0      �  �  �  �  �  �  ^  2    �  �  	u  	�  
D  
�  
�  
�  
�  
�  
c  
!  	�  	Q  �  �  �  �  �  �  �  B  �  	  	W  	�  	�  	�  	�  	�  	N  �  h  �    M  y  �  ^  �    	  	  
  
    �  �  �  �  �  �  �  �  �  �  �  �  k  Q    5  E  S  [  \  N  :  !  �  �  �  ]    �  P  �  V  �  8  �  �    "    �  �  �  V    
�  
|  
  	�  	-  �  �  <  <  �  M  E  >  6  .  &             �  �  �  �  �  �  �  �  �  P  Y  a  [  L  ;  %    �  �  �    �  l  "  �  �  :   �   �  =  f  c  Y  K  ;  )    �  �  �  q  -  �  |  !  �  ^  �  U  
$  
_  
�  E  �    U  s  �  �  e  ,  �  s  
�  
@  	N    �  {  I  m  �  �  �  �  �  �  [  #  �  �  G  �  �  7  �  2  e  G  �  �  �    c  D  )    �  �  �  �  �  L    �  �  j  ?  *  �  	  	#  	  �  �  �  Q    �  �  P  �  ~  �  c  �  �  �  "  Y  c  j  �  �  y  k  V  8    �  �  �  �  b  A     �  �  �  	y  	�  
  
N  
w  
�  
�  
t  
[  
1  	�  	�  	U  �  u  �  �  �  a  /    �  �  �  �  ]  (    �  �  �  �  �  �  t  3  �  �  c    �  �  �  �  �  �  }  |  z  i  U  A  .      �  �  �  �  �  �  �      �  �  �  �  �  k  D    �  �    =  �  �    �  �  �  �  �  �  a  ?  )  (  ,  !    �  �  �  q  =  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  l  a  U  �  �  �  �  �  �  �  �  w  q  p  s  {  ~  y  s  j  `  U  I  q  j  b  [  Q  A  0      �  �  �  �  �  �  �  s  `  M  :  <  I  3              �  �  �  [    �  J  �  C  �  	  Y  �  �  �  �  ~  ]  6    �  �  h     �  �  9  �  �  9  �  O  i  v  x  k  X  C  ,    �  �  �  O    �    @    �  �  	�  
e  
�    0  =    
�  H  x  [    
�  
  	�  �  /  C  4  �                        �  �  �  �  o  G    �  F  z  R  %  �  �  �  u  s  �  l  H    �  �  R  �    =  =  �  �  �  �  �  �  �  u  T  +  �  �  �  \    �  �  8  �  x  �  }  r  c  Q  >  +    �  �  �  �  g  <    �  �  K  
  �  p  �  �  �  }  a  B  "    �  �  �  y  V  3    �  �  �  Z  '  $  '  *  ,  -  +  (         �  �  c  	  �  =  �  _  �  S  )  (  '  !    	  �  �  �  �  ^  :    �  �  �  y  P  &  �  �  �  b  ;    �  �  �  j  =  
  �  �  =  �  �  C  �  l     �  �    6  [  s  z  q  Z  4  �  �  	  
l  	�  	  T  �  &  �
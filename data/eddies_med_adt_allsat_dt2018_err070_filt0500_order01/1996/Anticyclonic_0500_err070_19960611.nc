CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?° ě��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M׬�   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���
   max       =�G�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @E�Q��     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vw�
=p�     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q            x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @�S`          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >bM�      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-7�      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B-K      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >$�   max       C���      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�\   max       C���      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M׬�   max       P��U      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Z���ݘ   max       ?�j~��"�      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       >bN      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?5\(�   max       @E�Q��     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vw��Q�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q            x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @��@          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         BC   max         BC      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�    max       ?�h	ԕ+     p  S(   
                        #         K   M                  
      1            8         	   L      
         =            
                  
   3   +         o                        �      N��KN�QO�FO#EN:3TM׬�O: �N�J�O�NmmFNOcQP��O�c�O9:�N�[=N�(N��*O͒N��tN��!P&�lN�;O�On%O�vO� �N��GN��O�bnN�c	N��fN �]Oh~�O�!O �eO�!DO1dO�COE��N}~OP��O&�Nގ�NԌ�P�{P��O��<N�P9ùN�~zNi�N=�O��FOs
N+TDN@��O���N@�N�:���
�e`B�#�
�o��`B�ě���o:�o;�o<#�
<49X<D��<T��<T��<e`B<e`B<u<u<u<�o<���<�1<�9X<�9X<�9X<�9X<�9X<�9X<ě�<ě�<���<�`B<�<��<��=+=C�=t�=�P=�P=�P=��=��=��=�w=0 �=0 �=H�9=L��=aG�=aG�=aG�=ix�=ix�=ix�=���=���=��=�G�����������������������������������������B>CFN[`gktutphg[ZNBB�~~~���������������`Z_benszrnib````````����������������������������
�������)05@>:5.)"'08IUbb[UIJD<0 ��� 

�������5136;BOZWOB655555555���B[v�t[PE@5���ZV`g��������������gZ��������������������[[^gt�����tge[[[[[[[ #/<HOH<6/*#        ��������������������,)'+/3<HIUWWUQLH@</,fbefhist�������tohffrkoost�����������trr���5BNX[ZTJ>5)����������������������������������������������)5BHCB>5)$�������)4=;/) ���
)15BNQTUQNB5)
ab__bghtv|�|}wthaaaaz���������������zzzzHLSTm����������zmaIH��������������������MFINMN[]ghkjg][NMMMM�������������������� ��	)07<EEB8'���� )6:A@=6)�����������������������������
#/<II<#����[`eht���������the\[!#,08<@BDB<0#����
'/331/)#
����

�������"#/<?HILKHE</("//:;AHKTacgjea\TH;//,&$#)/<AHIKKHC</,,,,tnnrt���������ttttt�����6?>6,$����������� (-1.)����������������������&'/<?A><;/&&&&&&&&&&����-/-���������������������������������������������������������������������)5BMMIB7)!�!)5:BNPNNB5-)T[gt����tgf[TTTTTTTTUH</('/<HLUUUUUUUUUU���������

���������������������������������������Ź������������������ŹůűŸŹŹŹŹŹŹ�������������������������������������������������������������������������������ż��������������������������{�r�i�o�r������������ݽӽݽ���������'�(�3�;�5�3�'�#�$�$�'�'�'�'�'�'�'�'�'�'����������$�����������������������������������ܾ׾Ծ׾߾������r�������Ƽ����������r�Y�M�@�4�0�7�M�Y�r���������ľʾҾʾ������������������������z���������������z�v�u�y�z�z�z�z�z�z�z�z�#�I�}ŎŐ�|Ń�{�n�U�?�8�0����������
�#�B�O�[�`�b�f�g�[�O�B�6�)��������+�9�B���������ùϹܹ����ܹù������������������������������������ÓÔ×ÚÚÓËÉÇÅÇÐÓÓÓÓÓÓÓÓ���������ĿĿĿ��������������������������������������������������������һ-�:�F�S�[�_�l�m�l�l�_�S�F�:�2�.�-�,�-�-�Z�g�s�������������������s�g�Z�Y�V�Y�Z�Z�����0�3�0�(�����ѿ����������ѿ��¿����������¿³¶¿¿¿¿¿¿¿¿¿¿¿�
���
��#�/�<�C�I�C�<�8�2�0�.�#���
�f�s�z�v�s�w�w�t�f�Z�M�H�A�;�=�A�M�Z�^�f�tāčĚĢĪīĠč�t�h�B�6�)�,�6�B�[�h�t�����)�&���������������ûþ���������Z�f�s���������������s�f�Z�W�V�Z�Z�Z�Z�M�U�N�M�D�A�4�(�����(�4�A�F�M�M�M�MĴĿ������������ĿĳĦĚčā�s�q�zċĦĴìù����������������������ùõñìâìì�.�;�G�T�`�c�b�`�W�T�G�;�8�.�-�-�.�.�.�.�����������������������������������������!�-�:�S�]�S�F�B�6�-�!����������!�������	�/�A�C�?�5�/�������������������Óàìù��þùíìàÕÓÇÃ�|�zÇÊÓÓ��(�A�N�Z�^�c�N�<�:�-���������������T�`�m�p�w�m�k�b�`�Y�T�G�D�;�1�/�7�;�G�T��(�4�A�Z�`�f�o�����s�f�Z�Q�4�$����m�y�������������y�m�`�W�G�;�C�G�O�T�d�m�V�W�b�i�o�p�o�b�^�V�I�H�I�T�V�V�V�V�V�V�׾��	��"�.�"��	����׾Ͼʾ����ʾ��
��#�%�0�:�<�H�A�<�0�#���
�����
�
E�E�E�FFFFFFFE�E�E�E�E�E�E�E�E�E濒���������ĿƿʿĿ����������������������r�������ĺź��������|�e�L�G�>�;�<�F�L�r¿�������� ��������¦�t�o�m�t¦¿�����������������������������s�h�o�s����ǔǡǬǥǡǔǈǇǈǏǔǔǔǔǔǔǔǔǔǔ��������������������f�R�N�N�R�U�f�������������
����������������(�4�A�G�D�A�4�(��"�(�(�(�(�(�(�(�(�(�(�zÇËÇÃ�z�n�a�]�a�n�q�z�z�z�z�z�z�z�z��"�/�;�=�F�E�=�0�)�"��	���������	������������*�9�=�I�=�$���������������������	���	�	���������²±¯­²¿������¿³²²²²²²²²²D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~DvDtDtD{D��k�_�S�F�D�F�P�S�X�_�l�x���x�k�k�k�k�k�k�4�@�D�M�O�Q�M�L�@�5�4�'���'�/�4�4�4�4 & I 4 + K g 6 3 3 s ] ? Y : 9 � D ; 3 L - X d p < 8 M ^ 9 h 1 h y ) J J 0 � * c _ A " + , f  @ e m X i : m � � 0 [ (    �  $  *  ]  G  8  �  �    �  ~  �  �  �  �  �  �  Y    9  �  4  �  i  �  w    �  =    �  i  R    #  .  �  �  �  P  �  n  �  �  �    t  :  �  �  -  X    e  �  �  �  g  ˼#�
�t��D��;�`B�D����o;�`B;�`B=�w<�o<e`B=�-=�E�<ě�<�o<��
<���='�<���<��
=�7L<���=49X='�=��-=P�`<�h<�=���=C�=t�=+=D��=�j=y�#=�o=@�=<j=}�=#�
=q��=m�h=�7L=D��=�E�=� �=��=T��>$�/=q��=q��=�hs=��P=���=u=�9X>bM�=�->oB�B�YB�%B*B(%�B JB��B�XB&,�B�[BR@B�B�4B)�B	�JBv�BZB��BVB
SB�`B�(B��BP'B��B5�B��Bc�B &B��B��B!��B�B�8B"T�B1B�zB%��BKiB� BOA���B��B�rB�]B�(B9�BsBQ�B6�B�B�[B�{B&�B	]B��B_B-7�B��B��BŨB��B)�hB(;�B �:B�MB��B&'PB��BB�B�B�3B�B	�ZB�BJzB0�B��B
TB��B��BìB�BȵB@�B)�BE3B +#B1GB��B"4?B��B�B"1�B�zB@�B%?�Bk�B�^Bb)A��B��B�B�XB�B7�B��B<�B�sB7?BĀB<�B?�B	��B�B;�B-KBJuA�ȦAІA��N@��zA-�*?�b�A���AV=�@�AM��A�UnA�,�Aם�>$�A1?A���Au�A�s@�R�A��HA�1'A��"A��aA@	�A�ƠAѶMAC�rA8�A�s�A�R�Ac��@%
�@j�GA��A�p^A�/GAfn8A<�NAjn�B
�AWa�A�EC���At�@4+A��A�:BhX@��A3G!A8�uA��A�@<BՒA�^vA��C��b@�R%@��A���AЊ�A�t@���A-(�?�$�A���AU
�@�(ZAL�A��A�oA׃h>�\A0h�Aʈ�Au�A�|q@���A��PA��A��vA��A?�A�e�A�y�AE ;A8�%A���AΏAc-@$)�@o�4A�u�A�i�A���Af�uA:�AkgB��AU��A�_!C���Au +@��A�p�A���BI+@���A2��A8�PAȅ�A�IlB˺A�iDA��KC��'@���@���                           $         L   M                        1            8         	   L               >                                  3   ,         p                        �                                 +         ?   %                        +            !            %               #      '                           )   )         -               !                                                   =                           #                                             %                           #            -                              Np��N�QO�FN��tN:3TM׬�O: �N�J�O���NmmFNOcQP��UO&�6O9:�N�[=N�(N��*N�ՙN��tN��!O��N�;N�,!N��O82O(�%N��GN]�OA�`N�c	N[�JN �]Oh~�O���N�0�O��<O1dO�CO��N}~O6��O{VN���NԌ�O�i�Og��O��<N�P(�YN�~zNi�N=�O��FOV��N+TDN[�O�`N@�N�:  9  (  �  �  �  9  �  �  1  9  z    	�  /  �    @  �  ?    �  }  �  8  }  r  �  �  
<  S  �  }  �  e  A  �  (  �  �  �  �  �      o  g  �  �  �  j  �  �  V    �  w  
    ��C��e`B�#�
�D����`B�ě���o:�o<e`B<#�
<49X<�o=@�<T��<e`B<e`B<u<���<u<�o<�h<�1<���<�`B=49X<��<�9X<ě�=e`B<ě�<�`B<�`B<�=@�=�w=C�=C�=t�='�=�P=�w=�w='�=��=@�=u=0 �=H�9=e`B=aG�=aG�=aG�=ix�=m�h=ix�=��w>bN=��=�G�����������������������������������������B>CFN[`gktutphg[ZNBB��������������������`Z_benszrnib````````����������������������������
�������)05@>:5.)"! ,0<IUWZZXNI<0#!��� 

�������5136;BOZWOB655555555��B[t|~tNC5)��������������������������������������������[[^gt�����tge[[[[[[[ #/<HOH<6/*#        ��������������������+)+./7<CHSTOJH<<2/++fbefhist�������tohffrkoost�����������trr��	)5BNSSRND5)������������������������������������������)5;@85)�������
))/1.)%)5BCKMNNHB5)( ab__bghtv|�|}wthaaaa��������������������icelmz����������zsmi��������������������HKN[gihg[[[NHHHHHHHH�������������������� ��	)07<EEB8'����
).6:;:6)������������������������ 	#/<HH<#������[`eht���������the\[!#,08<@BDB<0#���
#/110/+%#
���

������� #&/6<CHKJHC</*$ 31;;BHLTabfiea[TH;33.'%%/0<?GHJJHA</....tnnrt���������ttttt������-551'����������""����������������������&'/<?A><;/&&&&&&&&&&�������+-*��������������������������������������������������������������������)5BMMIB7)!�")5=BJNOMMB?5+)!T[gt����tgf[TTTTTTTT*)/<HIJH</**********��������


�����������������������������������������������������ŹŹŶŹž�����������������������������������������������������������������������������������������������ż����������������������u�z�����������������ݽӽݽ���������'�(�3�;�5�3�'�#�$�$�'�'�'�'�'�'�'�'�'�'����������$�����������������������������������ܾ׾Ծ׾߾������Y�r�����������������������f�^�T�K�U�Y���������ľʾҾʾ������������������������z���������������z�v�u�y�z�z�z�z�z�z�z�z�I�{ŋō�zŁ�z�n�U�C�1�������������#�I�6�B�O�P�U�V�M�B�@�6�)������#�)�5�6���������ùϹܹ����ܹù������������������������������������ÓÔ×ÚÚÓËÉÇÅÇÐÓÓÓÓÓÓÓÓ���������ĿĿĿ���������������������������������������������������������޻-�:�F�S�[�_�l�m�l�l�_�S�F�:�2�.�-�,�-�-�Z�g�s�������������������s�g�Z�Y�V�Y�Z�Z�����(�+�*�&�������ѿ������Ŀݿ��¿����������¿³¶¿¿¿¿¿¿¿¿¿¿¿��#�/�<�@�F�A�<�6�/�&�#� ��������f�i�r�r�p�l�f�Z�M�J�G�J�M�Q�Z�_�f�f�f�f�h�tāčĔĚĝěĚĎčā�t�h�]�R�Q�[�_�h����������������������������������Z�f�s���������������s�f�Z�W�V�Z�Z�Z�Z��(�4�A�J�A�A�4�(�����������ĚĦĳĿ������ĿľĳĦĚĔčĉĄĆčĔĚìù����������������������ùõñìâìì�;�G�O�S�M�G�;�9�.�.�.�0�;�;�;�;�;�;�;�;�����������������������������������������!�-�:�S�]�S�F�B�6�-�!����������!�������	��%�/�6�8�3�/�"��	������������àìììùüúùìèàÓÇÄÂÇÓÙàà�A�N�Y�`�Z�N�;�9�,���������������(�A�T�`�m�p�w�m�k�b�`�Y�T�G�D�;�1�/�7�;�G�T��(�4�A�Z�`�f�o�����s�f�Z�Q�4�$����m�y�|���������~�y�m�`�^�T�G�C�G�I�T�`�m�V�W�b�i�o�p�o�b�^�V�I�H�I�T�V�V�V�V�V�V���	��%�"��������׾Ӿʾ��¾ʾ׾��
��#�$�0�9�<�G�?�<�0�#���
�� ��
�
E�E�E�FFFFFFFE�E�E�E�E�E�E�E�E�E濒���������ĿƿʿĿ����������������������e�r�������������������~�e�Y�P�F�B�C�L�e�������������	��������¿²ª¬²¹���������������������������������s�h�o�s����ǔǡǬǥǡǔǈǇǈǏǔǔǔǔǔǔǔǔǔǔ�����������������������f�U�Q�Q�Y�a�r���������
����������������(�4�A�G�D�A�4�(��"�(�(�(�(�(�(�(�(�(�(�zÇËÇÃ�z�n�a�]�a�n�q�z�z�z�z�z�z�z�z��"�/�;�=�F�E�=�0�)�"��	���������	�������)�8�=�E�=�-�$�������������������������	���	�	���������²¿����¿¿²²°®²²²²²²²²²²D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��k�_�S�F�D�F�P�S�X�_�l�x���x�k�k�k�k�k�k�4�@�D�M�O�Q�M�L�@�5�4�'���'�/�4�4�4�4 / I 4   K g 6 3 . s ] = = : 9 � D 4 3 L $ X > V   / M I " h ) h y $ M J 0 � ' c Z ? % + % G  @ c m X i : ` � X # [ (    u  $  *  �  G  8  �  �    �  ~  �  q  �  �  �  �      9    4    H  }  i    u  �    n  i  R  -  �    �  �  b  P  �  V  �  �  1  �  t  :  �  �  -  X       �  V  A  g  �  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC  BC      (  1  7  9  8  1  (    	  �  �  �  �  �  s  R     �  (  '  &  %  $  "                    
          �  �  �  �  �  �  w  n  e  T  C  1       �  �  �  �  �    \  a  g  l  r  y  �  �  �  l  N  +    �  �  �  }  U  +   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  b  J  3      9  >  D  I  O  U  Z  `  e  k  p  u  y  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  [  L  ?  2       �  �  �  n  6  �  �  �  �  }  m  ]  I  3      �  �  �  �  q  M  $   �   �  n  �  �  	  %  1  *    �  �  �  �  t  j  3  	  �  �  �  �  9  &    �  �  �  �  �  �  �  x  k  ]  I  *    �  �  �  �  z  y  y  x  x  w  v  u  s  q  o  m  k  l  t  |  �  �  �  �  �    �  �  �  ]    �     �  s  A  *  �  �  �  n    I  �    Y  v  }  �  	4  	q  	�  	�  	�  	~  	b  	)  �  b  �  %    I    /       $    
  �  �  �  �  �  v  Z  L  >  '    �  �  �  �  �  �  �  �  �  �  �  �  |  v  r  n  i  e  a  \  X  T  O         �  �  �  �  �  �  �  �  �  �  �  �  �  z  y  w  v  @  6  ,  "         �  �  �  �  �  �  �  �  j  Q  8      Y  t  �  �  �  �  }  b  C    �  �  f    �  �  C  1  �  �  ?  3  '        �  �  �  �  �  �  �  �  �  �  �  �  ;        
    �  �  �  �  �  �  �  �  �  r  _  F  +     �   �  �  �  �  �  �  �  �  �  �  �  e  7  �  �  Y  �  �    H  Q  }  m  \  L  <  )      �  �  �  �  �  y  ]  @    �  �  �  8  V    �  �  �  �  x  j  \  ;  �  �  :  �  `  �  u   �   ~  �  �  �    $  3  8  8  7  +    �  �  �  �  f  <  �  s  �  �  �  �      $  [  |  |  n  R  !  �  �    }  �  �  �  �  .  +  7  J  Y  g  r  l  b  T  C  /      �  �  `  �    J  �  �  �  �  �  �  �  �  �  z  p  f  X  J  =  0  #    �  �  �  �  �  �  �  �  �  �  �  |  g  P  8    �  �  �  �  �  �  c  �  	V  	�  	�  
  
+  
:  
<  
:  
.  
  	�  	n  �  i  �  �  ^  �  S  B  1      �  �  �  �  q  X  D  #  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  f  M  2    �  �  �  n  ;    �  }  m  ^  O  @  0  !      �  �  �  �  �  �  q  V  :      �  �  �  �  �  �  a  ;        �  �  �  B  �  �  k    �  �    ;  X  b  e  a  Y  H  &  �  �  \    �  H  �  �     �  �  �    *  8  @  9  "     �  �  W    �  }    �  4  �  �  �  �  �  �  �    i  O  1    �  �  Z  
  �  ]    �  x    (            �  �  �  �  �  �  �  �  q  N  #  �  /   �  �  �  �  �  �  �  �  �  �  �  �  r  Z  D    �  �  x  @    �  �  �  �  �  �  �  �  u  >    �  �  R    �  h    �  �  �  �  �  �  �  �  �  �  o  [  G  2      �  �  �  �  �  n  �  �  �  �  �    g  J  $  �  �  �  S    �  �  X    �  [  �  �  �  �  �  �  p  W  7    �  �  q  1  �  �  h  1  �    �  �       �  �  �  �  j  A    �  �  �  ?  �  �    �  �        �  �  �  �  �  �  �  �  �  u  Z  )  �  �  d     �    I  c  n  i  \  I  0    �  �  s  &  �  �  &  �  $  i  �  �  �  �  �  �    /  c  f  X  8    �  �  J  �  s    �  5  �  �  �  �  �  �  x  `  F  *    �  �  �  |  =  �  �  �  j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  i    �  O  �  �  �  m  
  
�  
  	~  �  �  P  �  �  j  h  f  d  b  _  Z  U  P  K  ?  -    	  �  �  �  �  �  r  �  �  �  �  �  �  �  �  �  �  �  y  n  b  V  m  �  �  �  �  �  �  �  d  �  �        '  /  2  5  7  :  =  B  H  N  V  V  J  =  -      �  �  �  �  c  3  �  �  |  D    �  �  &  �    �  �  �  �  �  �  �  �  i  Q  7    �  �  P    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Y  Q  W  s  f  N  -    �  �  �  �  {  `  D  )    �  �  �  u  �  0  q  �  �  �  �      �  \  �        �  �    
z      �  �  �  �  �  �  �  }  m  \  J  7  #     �   �   �   �      �  �  �  �  l  >    �  �  [    �  �  R    �  f  �
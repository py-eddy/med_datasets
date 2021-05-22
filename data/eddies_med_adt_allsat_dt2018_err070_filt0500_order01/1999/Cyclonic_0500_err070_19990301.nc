CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���E��     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�q{   max       P�V�     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <e`B     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @FNz�G�       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v@�\)       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @�O�         4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �$�   max       <D��     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�%   max       B0�;     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0��     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       < ��   max       C���     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C���     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          R     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�q{   max       Pu�v     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�hr� Ĝ   max       ?ѡ���o     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <e`B     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @FK��Q�       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v@�\)       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @��         4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F$   max         F$     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��S&�   max       ?ћ=�K^       cx               	      R   	                                 )                           
      
   2      	      
      )   	   "                                                               6   E   	         '                  3                  N��Np�eN�+N\��N�z9N�muP8�TO��O@�?O��YPt�O���N���N~��O^�Oa�~O9W�O֧�P�V�O�72N�LN��"N�<�NX#*NM6zNJ�LOU�No�fN�Z'N`p�O��1OĊ;N�(N�Q1N�[O3wOѺ�N��3O�_�O���N��Oo#N�'M�q{O��N7��Pi`N�TN3Q�P��O,a�O0��O�1�N�ԔOCpBP`N��O"�jN�?P՞P�&Nn�vNC��O/�~O��%NG�+O!c�O�h�N��M�4O���N��O_�]N�EO�)O�&�N�p3<e`B<#�
<#�
<o;ě���o�o�o�o���
�ě���`B�o�t��t��T���T���e`B�e`B�e`B��o���
��1��1��1��9X��9X�ě����ͼ��ͼ�`B��h��h��h��h�����o�o�o�+�C��C��t��t��t���P��P��P��w�',1�,1�8Q�8Q�@��@��@��@��H�9�H�9�L�ͽY��Y��Y��aG��e`B�ixսm�h�q���u�u��%��o��+��+����/#"#'0;820/////////������������@BORP[^hiha[ROB?==>@46@BCORZWOIB86444444 #+/<=HOKHC<40/#    ��������������������BNt������[B5)!#/2<AHLH<1/#"#%/<HUbb]VUH</#"���
#/7;770"
�����$/Hbz����znUH/#~�������������}���~##04<ADD<<0########��������������������������������������������������������������������������������)2BO[cnehtqsqh[B6-+)#0<I~�������{U<0����������������������������������������xz�����������zxuxxxx����������������������������������������
#&#"
	���
�������������
 ###"
��������������������������$)6=:6))COO[hqoh[OCCCCCCCCCC�������������������������"������������������������������U[]]bghtxywtpkh[UUUU#/09920#�����������������������&'$���������\aamwz||zxrma^[Y\\\\��������������������*6COW\glmh]QC6*'$�����������������������""%,/-'"	����^agmz��������zmka`^^;;CHOJHH;:;;;;;;;;;;����)36) �������BBKO[\[ONB87BBBBBBBBY^gt������������yt]Y[anpqnfa\X[[[[[[[[[[��������������������Ubn{���������{naUQNU#%0<IKORSOIE<0&#����������������������������������������./<DHQU\ad`UPHE<3/-.fnqtz���������zuqnkf`t�������������iia^`46BINORSOBA862444444�����

��������#0850'#���������);;)�����)B[iqm]N����������������������������agt����tggaaaaaaaaaa��������������������Ranz�������zpnaXRNLR���������������������

������!$"
������������������������45ABENQNEB8544444444��$*)',)������.5@BBLLB53..........����������������������������������������[gt����������tg[UUV[��������������������//26<HHQTUNH<0/.////�H�=�;�/�(�"��"�/�;�H�H�H�H�H�H�H�H�H�H�����������������������������������������'����'�2�4�?�@�A�M�Q�Y�\�`�Y�M�@�4�'�a�_�U�S�U�]�a�n�zÄ�z�u�n�e�a�a�a�a�a�a�B�@�6�3�5�6�=�B�O�R�[�g�[�O�L�I�B�B�B�B�����������!�-�5�-�-�!��
������ѿĿ��������������Ŀݿ�� �������ݿѾ������ݽ׽ڽݽ�������������(�!�(�*�*�(�,�4�A�M�Z�[�_�`�_�W�M�A�4�(àÕ×ÝØÔÔàìù��������������ùìà���������������������
�#�#�)�,�*������Z�U�f�p�s������Ǿ;̾ξѾо������s�f�Z�������������!�!�*�"�!���������	������)�4�5�)����������t�l�h�[�Z�U�Y�[�h�tāĊčĒđčąā�t�t�����x�q�x�������������ûܻ�ܻлȻ������H�A�<�1�0�4�<�H�U�Z�a�n�q�v�v�s�i�a�U�H�;�"���#�.�G�T�Z�`�m�y�����������m�G�;�����p�U�I�A�(��(�Z���������������������������������Ŀѿ�����������ѿĿ��������������������������������������������������� ����)�3�*�)� �����������������������������������������������g�e�^�g�s�������������s�g�g�g�g�g�g�g�g�������������������������������������������������������������������������������������������������þʾ׾������׾ʾ����n�c�l�n�{ŇŔŜŔŊŇ�{�n�n�n�n�n�n�n�n�ܹԹϹιϹйչܹ������������ܹܹܹܹϹùù����ùϹع۹ٹϹϹϹϹϹϹϹϹϹϻ:�-�!���&�-�_�l�x�������������l�_�F�:�ѿ����������Ŀѿݿ���������������
�������%�*�1�.�*��������нʽĽ������������������Ľнҽѽнннкɺº��������ɺֺ�����ֺɺɺɺɺɺɻ���߻ܻٻܻ�������������������������������
��'�3�;�E�H�C�;�#��
��������ƹ�������������������������������̼Y�W�M�1�1�=�M�Y�f�r��������������r�f�Y�ʾ��������������ʾ׾�������	���׾��h�\�P�O�D�O�Q�\�h�uƁ�|�u�h�h�h�h�h�h�h��	���������������������	��$�.�1�/�"��������������������������������������������������!�����������������������������������������������������������������������������������������������������������������5�B�K�F�5�1�������������������������������������������ֺӺֺں���������ֺֺֺֺֺֺֺֻû��������ûܼ�'�@�f�q�h�c�U�4����лþ������������(�4�>�A�E�@�4�.�(������ż��������������*�/�*����������[�O�@�<�C�S�U�\�uƎƚƥưƶƳƧƎ�u�h�[�������������������ùϹԹϹȹù����������Ϲ͹ù��������������ùϹ�� �����޹ܹ��0�� ����������"�I�U�nłńŀ�{�n�b�<�0���������������û̻ƻû������������������Ľ������������Ľнݽ���������ݽнĽĽ�������������������������������������������ĳčā�h�O�8�B�G�[�tāđĦĳ����������ïçååâìù���������)����������ּ̼˼ּ�����������������ݼּּּ־����������	���	�	���������������������	� ����������	��"�.�7�8�3�.�,�"���	�Ŀ������������Ŀѿ����������ݿѿĺ����~�r�e�_�e�r�~�����������������������k�k�l���������ûλлܻ�ܻлû������x�k�.��������!�.�:�S�d�u�y���y�p�`�S�G�.�� �������!������������ŔœŇŇŇōŔŚŠŢŠśŔŔŔŔŔŔŔŔ���j�k�s���������������������������������������������������������������������������߹�����'�@�Y�\�Y�L�@�3�'��������������������������ü�����������������¤¦²��������� ������������¿²¤�0�$������������$�0�=�I�c�m�h�_�I�=�0D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� U M N Q D ] O @ = / 8 @ 4 A % K / E > F Q ; 3 Q C : 8 X ; f R F  n " 0 N N ' 1 O ` 1 z F X : = ` \  m F  D E W & = y n j _ N 1 � z a U : 2 > s J Y W 4    O  �    �  �  �  �  +  �  2  ~  P  �  �  4  �  �  �  .  -  ?  �  �  t  q  J  >  �  �  �  �  �  �  �  �  �    �  G  7  �  G      -  V  �  ,  �  )  i  �  �  ,  �  �  �  \  ,  �  �  �  �  �  }  �  �  �      �  D     �  �  [  �<D��;�o�t��o�o��`B����#�
��1�C������D���������P�,1�o�]/���ͼ�1�C���/������j�����<j�C��0 Ž\)���-�u��P�+��w�@���t��#�
����u�#�
�L�ͽH�9���<j�'q���#�
��w��O߽�7L�m�h��+�e`B�}󶽝�-�q���ixսP�`���`����q���m�h��C���j�q��������1��%�}��S���%������{���罶E��$�B%��BbB�BclB��B+��B�eBR�B�OB�B�jB ~�B%�BB��B�XB M9B!�B�/B'�KB*#�B4�B [zB�"B�BYB9BB�@B �VB
�B�B�0B��B-�B�UB%@*B��Bo�A�O�B ǶB0�;Bd�A�%A�CjA�_�B �B� B
�BeJB�NB(ӵB&&B܅BAB-=BǞB�UB>~B#�B%w�B�B�JB,�.B	�B�B��B>B$'B�!B#�<B�BN�B�lB!ļB�B
!�B��B�,B%�B`2B�BA�B��B+�GB��By�B:�B��B:~B �B%�PB��B��B <B ��B�B(��B)�%B@�B AB�5B��BIBD�BB ��B�B= B�DB��B@�B� B%B�B,BF�A��B ��B0��B>�A���A�n�A�{B�RBB{B�eB:�B�LB(ĥB&@B�hBB�B�B��B7~B@�B#��B%I�B�EB��B,�zB	ܦB��B!B>�B$B�BK�B#�4B�HB�&B�}B"�B�
B	��B@IB��A���A���@�@A�B@A��@^��A|a�A/�]A;C#A�e�A�.�AI�AA
orA��A�s�@��lAŀ	Ag�>A���A|btA�C�AԼA���A��tA�eXAI�AP�.A�œ>��Z>�
�@�óA}��A�;�A$�a@:�G@�QKA�R�BӜ@�qPAT�B4A��@A��EA���A�Q�A�ԘA��1A�]2@Dɸ@�.�A5��A���B< ��>��A�c@��EA)k�A"�KA�m�AЧ�A�%AYێA]��A|�@9�@�W�A�@@��gA�VA���A���?�:�@�@A���B
=�C���A���A�~�@ԅ�AƉ�A�a|@b�A}�A-�	A;ÀA��mAI&3A	�AՆAܛ@��RA�Q�Al�VA��?AziA�oZA�V�A�e'A��}A�E�AI�AP�A�M>��4>�ϟ@��A{hA�F�A$��@9��@��5A���B��@�1�ASLpB�YA��RA�/A���A�|�A���A���A��@G��@�3�A6�yA�cBGoC���>��A�O@��YA)wA"B�A�&oA�v9A"AZ A]5TAz��@�8@�?EA�@��xA�{A�ƣA��?��Q@� �A���B?�C��H               	      R   	                                 )                                    2       	            *   	   "                                                               6   F   
         (                  4                                       )            '   '                  '   9                                    !   !                                             +         -         !         1            3   +                     !         #      !                                 '            %                     '   5                                       !                                             '         -                  /            3   )                     !               !            N��Np�eN}rWN\��N�z9N�muP�N?�OX/O`�O�_�O�1�N���N��N�@�Oa�~N���O֧�Pu�vO�72N�LN��"N�<�NX#*NM6zNJ�LN�F8No�fN�Z'N`p�OV�OĊ;N�(N�Q1N�[O3wOѺ�N��3O�_�O���N��Oo#N�uM�q{O��N7��P `>N�TN3Q�P��O��O"�O��sN�ԔO0�PU}ON��O"�jN�?P�P)�Nn�vNC��O/�~O.��NG�+O�O�%�N��M�4O��#N��O_�]N�EO�)O�&�N�p3  T  �  �  P  M  �  
�  �  �  �  -  �  �  �  R  �  �  �  ^  �    �  �    \  �    �  �  �  I  �  D  h  �  8  U  �  -  �  �    D  1  �  ,  k  �  �  j  o  �  J  �    {  X  �  <  	V  
�  u  H  !  R  �  �  �    t  	  �  i    �  �  	
<e`B<#�
;��
<o;ě���o�D���ě��ě��t���`B�49X�o�D���T���T���ě��e`B��o�e`B��o���
��1��1��1��9X�����ě����ͼ��ͽ#�
��h��h��h��h�����o�o�+�+�C��\)�t��t��t�����P��P��w�49X�0 Ž8Q�8Q�<j�D���@��@��@��T���T���L�ͽY��Y���+�aG��ixսq���m�h�q������u��%��o��+��+����/#"#'0;820/////////������������ABOV[b][OB?@AAAAAAAA46@BCORZWOIB86444444 #+/<=HOKHC<40/#    ��������������������$BN[t�������[B5)"#)'/<>F<1/##""""""" #$'/<HU[^XUKH<5/#  ��
#/1643+
������%/Hax���}�ynUH/#��������������������##04<ADD<<0########��������������������������������������������������������������������������������)2BO[cnehtqsqh[B6-+)#0I��������{U<0 #����������������������������������������xz�����������zxuxxxx����������������������������������������
#&#"
	���
������������
!
����������������������������$)6=:6))COO[hqoh[OCCCCCCCCCC�������������������������"������������������������������U[]]bghtxywtpkh[UUUU#/09920#�����������������������&'$���������\aamwz||zxrma^[Y\\\\��������������������,6CO\fjkf\PC6*(%!�����������������������""%,/-'"	����_ahmz������zmlaa____;;CHOJHH;:;;;;;;;;;;����)36) �������BBKO[\[ONB87BBBBBBBBZ_gt������������|t^Z[anpqnfa\X[[[[[[[[[[��������������������Ubn{���������{naUQNU##0<IINQQMIB<60)#" #����������������������������������������./<DHQU\ad`UPHE<3/-.wz�����������zvrnnrwakt������������zgc^a46BINORSOBA862444444�����

��������#0850'#��������'99)������)B[foj[N4��������������������������agt����tggaaaaaaaaaa��������������������^anz�������znea\WW^����������������������

������ #!!�����������������������45ABENQNEB8544444444���"'%%�������.5@BBLLB53..........����������������������������������������[gt����������tg[UUV[��������������������//26<HHQTUNH<0/.////�H�=�;�/�(�"��"�/�;�H�H�H�H�H�H�H�H�H�H�����������������������������������������4�/�-�4�9�@�M�N�Y�Y�M�@�4�4�4�4�4�4�4�4�a�_�U�S�U�]�a�n�zÄ�z�u�n�e�a�a�a�a�a�a�B�@�6�3�5�6�=�B�O�R�[�g�[�O�L�I�B�B�B�B�����������!�-�5�-�-�!��
������Ŀ��������������Ŀѿݿ����	�����ݿľ�����������������������M�D�A�4�1�.�,�0�4�A�M�T�Z�[�[�Z�W�Q�M�MÞàáÜÙÜàìù����������������ùìÞ���������������������
��#�(�*�)�������s�k�n�t�t�y����������þȾȾɾɾ�����������������!�!�*�"�!����������
���)�/�,�)������������h�_�[�Z�[�]�h�tāąčċā�t�h�h�h�h�h�h�����x�q�x�������������ûܻ�ܻлȻ������H�H�=�<�:�<�H�J�U�a�h�i�i�a�_�U�H�H�H�H�;�"���#�.�G�T�Z�`�m�y�����������m�G�;�v�Y�D�(�(�9�Z�������������������������v�������������Ŀѿ�����������ѿĿ��������������������������������������������������� ����)�3�*�)� �����������������������������������������������g�e�^�g�s�������������s�g�g�g�g�g�g�g�g�����������������������������������������������������������������������������������������������ʾ׾ݾ����׾ʾ��������n�c�l�n�{ŇŔŜŔŊŇ�{�n�n�n�n�n�n�n�n�ܹԹϹιϹйչܹ������������ܹܹܹܹϹùù����ùϹع۹ٹϹϹϹϹϹϹϹϹϹϻ:�/�/�:�>�F�_�l�x�����������x�l�_�S�F�:�ѿ����������Ŀѿݿ���������������
�������%�*�1�.�*��������нʽĽ������������������Ľнҽѽнннкɺº��������ɺֺ�����ֺɺɺɺɺɺɻ���߻ܻٻܻ�������������������������������
��'�3�;�E�H�C�;�#��
��������ƹ�������������������������������̼Y�W�M�1�1�=�M�Y�f�r��������������r�f�Y�ʾ������������ʾ׾�������	����׾��h�\�P�O�D�O�Q�\�h�uƁ�|�u�h�h�h�h�h�h�h��	���������������������	��$�.�1�/�"��������������������������������������������������!�����������������������������������������������������������������������������������������������������������������&�5�>�B�5�/�������������������������������������������ֺӺֺں���������ֺֺֺֺֺֺֺֻû��������ûܼ�'�@�f�q�h�c�U�4����лþ���������(�4�<�A�C�A�=�4�*�(������ž����������������%�����������h�b�O�M�F�O�\�h�uƁƎƚơƭƳƧƚƎ�u�h�������������������ùϹԹϹȹù������������������������ùϹ�����������ܹϹù��0�#�������������I�U�uŀł�{�k�b�<�0���������������û̻ƻû������������������Ľ������������Ľнݽ���������ݽнĽĽ�������������������������������������������ĳĚā�h�[�@�G�I�[�tāďĦĲ����������ùðèæèäìù�����������������¼ּ̼˼ּ�����������������ݼּּּ־����������	���	�	���������������������	� ����������	��"�.�7�8�3�.�,�"���	�Ŀ����������Ŀѿݿ�������������ݿѿĺ����~�r�e�_�e�r�~�����������������������x�l�k�x�����������û̻лܻлû��������x�.��������!�.�:�G�S�a�l�q�y�j�`�S�G�.�� �������!������������ŔœŇŇŇōŔŚŠŢŠśŔŔŔŔŔŔŔŔ�����p�n�s�������������������������������������������������������������������������߹�����'�@�Y�\�Y�L�@�3�'��������������������������ü�����������������¤¦²��������� ������������¿²¤�0�$������������$�0�=�I�c�m�h�_�I�=�0D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� U M / Q D ] M r B 1 5 ) 4 :  K 0 E < F Q ; 3 Q C : 3 X ; f H F  n " 0 N N ' 0 O ` ' z F X 7 = ` \  g A  D H W & = z l j _ N  � | a U : 4 > s J Y W 4    O  �  �  �  �  �  �  |  <  �  Z  �  �  1  �  �  �  �    -  ?  �  �  t  q  J  �  �  �  �  �  �  �  �  �  �    �  G    �  G  �    -  V  V  ,  �  )  1  �  7  ,  �  �  �  \  ,  B  T  �  �  �  m  �  �  A      �  D     �  �  [  �  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  F$  T  P  K  G  C  ?  ;  7  3  /  .  1  3  6  8  ;  >  @  C  F  �  �  �  }  m  ]  N  >  /         �  �  �  �  �  �  �  �  �  N  h  ~  �  �  �  �  �  r  F    �  �  a     �  �  D  �  P  W  ]  `  b  ]  W  P  E  8  )      �  �  �  e  5    �  M  6        �  �  �  �  �  �  c  >    �  �  �  f  1   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  ^  J  2      
E  
r  
�  
�  
x  
^  
7  
  	�  	w  	#  �  W  �  K  �    8    �  �  �  �  �  �  �  �  �  �  �  �  �  �        #  1  H  _    �  �  �  �  �  �  �  �  �  �  q  S  5    �  �  i    �  �  �  �  �  �  �  �  �  �  x  R  !  �  X  �  �  �  �  ~  c  !  %  �  �  �  P  +    �  �  �  L  ,  :    �  �  �  \  c  �  �  �  �  �  �  �  h  J    �  �  t  <    �  �  �  T   �  �  �  �  �  �  �  �  �  �  {  o  ]  K  :  (       �   �   �  b  i  o  t  y    �  �  �  �  �  �  �  �  �  �  �  �  �  y       =  O  Q  L  C  7  '    �  �  �  W    �  u    �    �  �  �  �  �  �  �  �  �  �  �  �  {  M    �  =  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  S    �  �  4  �    �  �  �  �  �  �  �  �  �  �  �  ~  [  9    �  �  `  *   [  ^  ^  [  N  =  +    �  �  �  s  (  �  �  i  @    �  �   �  �  �  �  �  v  c  O  9    �  �  �  �  l  L  =  )     �   �        �  �  �  �  �  �  �  �  �  ~  m  [  I  5  !    �  �  �  �  y  T  0  
  �  �  �  o  I  ?  !  �  �  �  g  4     �  �  �  �  �  �  �  �  s  ^  J  8  %    
  �  �  �  �  �                �  �  �  �  �  �  �  �  �  �  �  �  �  \  W  R  M  H  C  =  8  3  .  &         �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  {  g  S  ?  +    �  �  �  �  �          �  �  �  �  �  y  N    �  �  F    �  �  �  �  �  �  p  K  $  �  �  �    T  +     �  �  �  �  �  �  y  �  �  �  �  �  �  �  k  P  3    �  �  �  �  s  M  $  �  �  �  �  �  �  �  �  �  �  �  �  �  }  w  p  h  _  P  @  /    �    '  >  F  H  F  B  8  #    �  �  L  �  M  |  �  J   �  �  �  �  �  �  �  ~  m  X  <    �  �  n    �  M  �  �  k  D  @  =  5  *      �  �  �  �  l  G  "  �  �  �  �  �  y  h  \  O  C  6  +  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  f  V  D  3  !    �  �  �  �  �  a  %  8  4  .  $      �  �  �  �  �  �  g  &  �  �  <  �  �  b  U  2    �  �  �  O  %  �  �  W     �  R  �  �  <  �  �  V  �  |  o  a  R  B  ,    �  �  �  �  �  r  S  )  �  �  �  {  -       �  �  �  �  �  �  �  �  m  O  -    �  �  �  ~  �  �  �  �  �  �  �  �  �  �  �  �  {  [  8    �  �  5  �    �  �  �  y  i  Z  I  8  '    �  �  �  �  �  t  W  ,   �   �    �  �  �  �  �  f  B    �  �  �  j  :  �  �  a    �  �  ,  ?  ?  8  1  (      �  �  �  j  :  
  �  �  q  �  5  �  1       �  �  �  �  �  �  �  �  q  Z  D  .       �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  W  ;     �  ,    	  �  �  �  �  �  �  �  u  b  L  3       �   �   �   �  _  k  g  Z  B  '    �  �  �  S    �  �  �  ~  B  �  �  P  �  �  �  �  �  �  �  �  �  �  �  �  �    u  j  `  V  L  A  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  p  g  ^  V  M  j  f  ]  R  ;    �  �  �  }  b  A     �  �  H      �  �  T  `  l  m  d  W  B  )    �  �  �  [    �  >  �    �    �  �  �  �  �  i  =    �  �  i  .  �  �  �  T  0      �  )  2  D  F  ?  :  8  4  ,    �  �  �  �  C  �  �  ,  �   �  �  �  �  �  �  �  f  C    �        �  �  ,  �  S  �  U            	  �  �  �  �  �  �  ]  )  �  �  |  J    V  v  w  i  P  2    �  �  �  W    �  �  �  �  K    �  �   �  X  T  K  :  '    �  �  �  �  �  �  �  z  \  :    �  �  �  �  �  �  �  �  �  �  |  r  c  O  5    �  �  �  �  t  `  M  <  7  2  -  (  "          �  �  �  �  �  �  �  �  �  �  	U  	V  	F  	   �  �  F  t  ?  9  �  r    �  Y    �  �  �  �  
�  
�  
�  
n  
(  	�  	�  
$  
-  
  	�  	w  �  g  �    <  )  �    u  w  y  t  o  j  e  _  Y  Q  I  @  6  ,  "        �  �  H  :  -        �  �  �  �  �  �  �  �  |  m  c  Y  O  E  !  !        �  �  �  �  t  H    �  �  �  �  �  �  u  q  g  �  �    4  I  R  O  E  7    �  �  �  :  �  i  �  �  /  �  �  �  �  �  �  �  r  b  S  D  7  +         �   �   �   �  �  �  �  �  v  `  R  X  [  K  :  %    �  �  a  #  �  �  }  }  �  �  z  _  @    �  �  �  \    �  �  :  �  �  Z  �  g          �  �  �  �  �  �  �  �  �  [  ,  �  �  �  @   �  t  m  e  ^  V  O  G  >  5  ,  "      
  
            �  	  	  	  �  �  �  �  {  ?  �  �  E  �  q  �  S  �  d    �  �  �  �  �  �  �  �  �  s  `  N  ;  *        �  �  �  i  P  9  $    �  �  �  �  �  �  �  �  �  �  z  e  =    �    �  �  �  a  ,  �  �  _  
  �  I  �  t    �  H    �  �  �  �  �  �  �  v  \  >    �  �  �  �  �  p  S  &  �  �    �  �  �    X  %    8  N    �  ~  �  �  �  *  W  �  2   �  	
  �  �  �  a    �  �  Z    �  e    �  A  �  x    �  l
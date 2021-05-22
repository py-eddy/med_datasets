CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�� ě��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       Pu�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�1       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F7
=p��     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @vk33334     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P`           �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @�-`           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �   max       <#�
       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0:0       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��4   max       B0@�       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       <�TO   max       C�&       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >A�   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          I       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P.��       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�A [�7   max       ?��_o�        @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�t�       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F.z�G�     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vjz�G�     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P`           �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @�r            XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?v�+I�   max       ?��c�A \     �  Zh         
               	            
      C   I                              .                  (   "            *                                                E         
   
                  ,               N���N~�uN�_O��OF�O&xN��M���O���NeP�O��O�$Pu�O<�OJOӀO>�O���NQOke�OV�O,TO{�XO��O�N0@O
I.N���O���P	FAP��O�<�O�MOeU�O�yoNh�N��%N#=GO�J�O@�.N��^O�*O90�N��N�O1�*O��N�_N�|�N�O�/�N���P.��N��N�UVNar�NO��N���N7��O��dO!��O��N��LO*M�4N�x<�1��o�D����o���
�ě���`B��`B�o�#�
�D���T���u��C���t����㼛�㼣�
���
���
��1��1��9X��9X��9X��j�ě����ͼ��ͼ��ͼ��ͼ���������/��h��h��h�����������o�o�+�+��P����w�#�
�#�
�,1�,1�8Q�8Q�@��L�ͽL�ͽe`B�e`B��o��O߽�O߽�O߽�O߽�1�������	������������������������������������������������������STX]amsuumkgfca_YTSS&*/6>CFLQ[OB6*����������������������'��������ABOPXQONB@AAAAAAAAAA�������������������������������������������������������������$&%"������)6]gke[OB6)�������##����������������������������GHQTW\abba^TJHF>@ADGMSXW]dt��������j`[OM��������������@IUakpz������znmfda@�������������������������������������������%(�������������
�����6=BOQ[mnjc^[OB@77306��������������������`bno{�����}{nbXYX[]`ot�����tlnoooooooooo����������������������������������������MN[gt�������tg[OKKMM�����

���������AHt����������t[NG?<A�����
$'/2."����������
����������������������������Tanz��������naZUMLMTu{{���������{wuuuuuu2;HMRQJHE;/122222222
!&










)5BNgkgb`[RNB5,'qw���������������thq�")00)�����������������������������������������������������������������?IQUbmnooonfbXUKI???ltv�������������wtolIL[n{�������{nbUI>?I�����������������������������������������������������������������!)55:65)����782%������#%
		)+5:>54)"!����������������������������������������������������tt����~|wtqlorttttttfgt�������������tgef����
	�������� #)/3<@BB=73/#" ���������������������%)/53)��9<GHIKH<;:9999999999ltw�����������tqllllùòùü����������������ùùùùùùùù�
��������
����#�#�#���
�
�
�
�
�
�H�G�A�H�I�M�U�a�d�a�`�^�]�U�H�H�H�H�H�H��ƺƳƧƟƗƚƧƳ����������� ������������	�������������"�(�.�2�4�3�.�"�àÛÕÕÚÓÓàæìùÿ��������ýùìà���x�z����������������������������a�`�`�a�n�zÃ�z�q�n�a�a�a�a�a�a�a�a�a�a�M�>�@�I�\�h�uƁƎƧƳ������ƵƓƎ�u�\�M��{�s�l�s��������������������������y�`�W�T�Z�`�y�������Ŀѿݿ��ݿ׿������������������ĿѿֿԿѿſĿ������������x�l�a�[�_�����������������������������x�������y�s�����������7�@�7����мʼ��ʼļ��������ʼּ����������ּ��A�;�5�4�5�A�S�Z�g�m�s���������s�g�Z�N�A���㾾���������������׾۾ھ���	����$����������������$�0�=�@�?�=�6�0�$�����������������������	�/�:�<�;�"������������
�������������������������	����������	��"�/�7�;�<�8�0�/�(�"��	�/�/�-�'�*�-�/�;�H�T�V�]�_�a�]�T�K�H�;�/�'���
��������������'�-�4�6�;�3�'�]�T�J�N�T�^�y���������������������y�m�]�����x�e�d�l�x���������ûл׻ܻܻ̻����������������Ľннݽ����� � ����ݽнĽ������������������������������������������U�N�I�=�F�I�U�b�n�t�{�~ŇŃ�{�v�n�b�U�UŔőŔŖŠŭŹ��������ŹŭŠŔŔŔŔŔŔ�׾ӾϾؾھݾ����	������	�����׾׾ʾþ����˾׾����	��%�,�0�*�"�	������������������������)�3�5�;�:�5�)���������������������������������
����������	���������	��"�.�3�;�A�F�;�.�"��ĥęĐĘĚĦĳĿ������������������Ŀĳĥ�����{�x�w�z���������ùϹ���޹ù������'�����'�0�4�@�@�C�@�=�4�'�'�'�'�'�'čĈĊčĚĦĳĴĳĮĦĚčččččččč�5�-�(�$�(�5�A�B�I�A�5�5�5�5�5�5�5�5�5�5���������������������*�/�5�,�������ŭŠŒŇŅ�~ŇŎŔŠŭŹź������������ŭ�5�-�/�3�5�:�A�N�X�Z�]�Z�U�N�G�A�5�5�5�5���������*�0�6�C�M�G�C�6�*�����Y�M�C�@�3�2�G�M�Y�f�������������r�f�Y�6�.�+�3�6�=�B�O�V�[�h�k�h�f�[�O�G�B�6�6�û������������ûлܻ�����ܻٻлû��z�p�m�a�`�]�]�]�a�m�z�����������������z�V�N�A�5�2�7�A�N�Z�g�s���������}�s�g�V�N�M�M�Z�b�g�s�{�����������������s�g�Z�N�A�A�;�;�A�I�N�Z�a�]�Z�V�N�E�A�A�A�A�A�A�g�f�f�g�p�s���������������������s�g�g�g�6�)� �����6�O�[�t�x�z��|�h�[�O�B�6ā�|āąčĎĚĦħĩħĦĚčāāāāāā���y�l�]�f�l�����Ľн��������н���!���!�.�:�@�G�S�[�S�G�:�.�!�!�!�!�!�!ùí÷ù����������������������ûùùùùÓÉÇÀÇÓàèéàÓÓÓÓÓÓÓÓÓÓ�x�w�m�x���������������������x�x�x�x�x�x���������"�$�*�&�$�������������������������������	����	��������������������ĿľĺĻĻĿ������������������������EiEfE]EYE\E`EiEtEuEwE�E�E�E�E�E�E�E�EuEiE�E~E�E�E�E}E�E�E�E�E�E�E�E�E�E�E�E�E�E������������'�(�(�'�$��������'�&�+�4�5�@�D�M�Y�f�k�f�f�^�Y�P�M�@�4�'D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��ù������������ùĹϹܹܹܹܹҹϹùùù� 8 . p � = E * R T u   3 L J G h a N M R . E k : Y ( : ; < 3 1 W $ Q - > 8 $ F 8 @ Y U N A 1 8 : X Y T ) A * i ` 1 G 7 � Y  R J c ^ -    �  u  �  �  �  C  �      o  l  9  �  #  �  �  W  �  S  9  �  �  �  	  �  I  :  1  �  	  Z  �  �  ,  �  �  w  �  5  �  �  �  6  �      �  *  %  �  �  �  �    �  �  k  a  �  �  M  c  _  �  9    �<#�
�ě��T���#�
��t���j���㼃o�o�T���'�9X��w���罸Q���ͽ49X�+�+��j��w�0 Ž<j�t���7L��w��`B��P�o�<j����q���@��#�
�0 Ž�\)�o�#�
�o�ixս49X�t��D���e`B�y�#��P�D���P�`�Y��8Q�D����S��L�ͽ�hs�]/�ixսe`B�e`B��7L�y�#���T��`B���罣�
���
��E��BۿB�4BS A�j�B0:0B�[BT5B��B�:B"D�B*�SB�	B�qB-:RB�A��ZBd�BF�B��B��B�kBBX�B�B!+4B(96Bz&Bz�B^�B	{�B�B	��B��B�B�B �B)3A���B�B��B
��Bi�B�"B B[BbB'fB
�OB(;B;BB�yBo�B��B�[Bk�B�[B�3B��B��Bf�B
�B��BqB��B�cB�EB�B�3B��BDA���B0@�B��B@�B�^B�RB"Y#B*��B�iB8&B-�B�`A��B@&B��B>�B{B�B@jB?OB��B ��B(?*B��B@ Bc�B	�B�,B
^�B��B�iBrB�FB)�A��4BiB��B
�hB��B��B :�B �GB'R�B
��B(C;BO�B>hB?�B@B�B�B~�B��B��B�4B� B>�B
Y�B�,B>�B�kB��B��B�5A�+A���A��BE@A\	A�LoAGv�AǃuBUAF��Ar.UAv�@��2A-A�0A��]AO��B	xA�43A1fA��A�(?�ˋAm�@�>�A*��A �FA�{A�k�AX�UAW�iA��A��HA^�4A�}s<�TO@˕JA��A���A�s!A��A�R�A��T@�{�A��@���A�r^A�рA�t�A�[�A�zAحEA�	�A$�A��A�[�A�Ў@���B	A�UrA�2�C��PC�&@�>F@�\pC��>>��A΋�A��JAņfB@�A[&hA�AG\A�iB�AD�-An�#Av�@�(�AA��A���AOsB	��A��WA1��A�vA��X?� �An��@���A*ʊA ��A�"A�xxAX�YAV��A�c�A�>�A](�A�?C���@�V�A߈�A�>:A���A���A�q$A��
@��CA؅e@�&HA��qA��ZA�)	A�n1A�A׀A��A ��A/�AχBA�c@��JB	,A�	�A�RC���C�'n@���@։4C���>A�                        
            
      C   I                              .                  )   "            *                                             	   F   	      
                     ,                                          !      )      !   ;         )      !                  !                  %   )   #         #                                                      /                                                                        %      !   %               !                  !                     !                                                                  /                                       N'�%N~�uNkEKO��O/	7N��dNC�M���O�bNeO��O��O�$P�FO)��OJO9M�O)��O���NQOke�OG&O]	Ob mO��O�N0@N���N���O\bOĬ�O�ܝO��-O�MOeU�O���Nh�N��%N#=GO�J�O@�.N���O�*O(�pN��hN��O1�*O��N�_N�|�N�OgN���P.��N��NZ��Nar�NO��N���N7��O��dN��,N�#�N��LO*M�4N�x  +  �  �  �  �  f  [  j  :  ;  {  �  |  �  �  V  &  �  @  F  =  I  
  �  �  �  ~  M  {  .  I  W  �  �  V  M  �  �  �  �  �  �  h  �  	�  Q  ^  �  b  z      �  2    I  6  n  �  F  �  
]  5  �  �  �  �<�t���o��o��o�ě��49X�49X��`B�D���#�
�e`B�T���u��P��9X�����/��1���
���
��1��9X�ě���j��9X��j�ě���`B���ͽo�+�o��h��/��h�\)��h���������o�o�+�C��C���P����w�#�
�#�
�}�,1�8Q�8Q�D���L�ͽL�ͽe`B�e`B��o���������O߽�O߽�1������� ������������������������������������������������������STX]amsuumkgfca_YTSS*06BCJORTTC=6*���������������������������������ABOPXQONB@AAAAAAAAAA�������������������������������������������������������������$&%"������)6]gke[OB6)���������������������������������GHQTW\abba^TJHF>@ADGeht���������thfb_^`e��������������@IUakpz������znmfda@���������������������������������������������$'�����������	�����16@BOO[lmib[OB=88631��������������������`bno{�����}{nbXYX[]`ot�����tlnoooooooooo����������������������������������������S[]gtt�����xtjg[VRSS������
����������GNSg����������g[NFDG����
!)-)#
�����������
����������������������������SZanz�������~na^UOOSu{{���������{wuuuuuu2;HMRQJHE;/122222222
!&










)5BNgkgb`[RNB5,'qw���������������thq� )//)������������������������������������������������������������������EIUZbgkklbUNIDEEEEEEltv�������������wtolIL[n{�������{nbUI>?I������������������������������������������������������������������!)55:65)����782%������#%
		"#%))58<52)����������������������������������������������������tt����~|wtqlorttttttfgt�������������tgef�����


���������!#/<<=<:41/#!  !!!!���������������������%)/53)��9<GHIKH<;:9999999999ltw�����������tqllllùøù������������������ùùùùùùùù�
��������
����#�#�#���
�
�
�
�
�
�U�H�H�C�H�J�N�U�a�d�a�_�^�\�U�U�U�U�U�U��ƺƳƧƟƗƚƧƳ����������� ������������	����������	��"�+�.�0�2�1�.�"�àßÙÙßàìòù��þùñìàààààà������|��������������������������������a�`�`�a�n�zÃ�z�q�n�a�a�a�a�a�a�a�a�a�a�V�E�E�N�_�h�uƁƎƧƷƺƯƧƚƎƁ�u�h�V��{�s�l�s��������������������������y�m�`�X�W�]�`�y�������Ŀοۿۿο��������������������ĿѿֿԿѿſĿ������������x�l�a�[�_�����������������������������x���������������������+�+�&���ּ����ʼǼ��������ʼּ�������������ּ��A�;�5�4�5�A�S�Z�g�m�s���������s�g�Z�N�A�����������������ʾϾ׾����׾ʾ������$��������������$�0�=�@�?�>�=�4�0�$�����������������������	�/�:�<�;�"������������
�������������������������	����������	��"�/�7�;�<�8�0�/�(�"��	�;�2�/�.�+�(�*�.�/�;�H�T�U�\�^�`�\�T�H�;�'� ���� �����������'�,�2�4�8�3�'�m�`�T�O�R�T�a�y���������������������y�m�����x�e�d�l�x���������ûл׻ܻܻ̻����������������Ľннݽ����� � ����ݽнĽ������������������������������������������b�_�U�I�C�I�M�U�b�n�{�}�{�z�n�d�b�b�b�bŔőŔŖŠŭŹ��������ŹŭŠŔŔŔŔŔŔ�����������	�������	�����׾ʾž¾þʾ׾������#�&�"��	���������������������������#�.�4�6�3�)�������������������������������� �����������	���������	��"�.�3�;�A�F�;�.�"��ĥęĐĘĚĦĳĿ������������������Ŀĳĥ���������~�}�����������ùϹ���عù����'�����'�0�4�@�@�C�@�=�4�'�'�'�'�'�'čĈĊčĚĦĳĴĳĮĦĚčččččččč�5�-�(�$�(�5�A�B�I�A�5�5�5�5�5�5�5�5�5�5���������������������*�/�5�,�������ŭŠŒŇŅ�~ŇŎŔŠŭŹź������������ŭ�5�/�0�3�5�;�A�N�V�P�Q�N�E�A�5�5�5�5�5�5���������*�0�6�C�M�G�C�6�*�����Y�U�M�D�@�4�3�H�M�Y�_�f�������r�m�f�Y�6�0�,�3�6�>�B�O�T�[�h�j�h�e�[�O�F�B�6�6�û����������ûлܻ��ܻԻлûûûûû��z�p�m�a�`�]�]�]�a�m�z�����������������z�V�N�A�5�2�7�A�N�Z�g�s���������}�s�g�V�N�M�M�Z�b�g�s�{�����������������s�g�Z�N�A�A�;�;�A�I�N�Z�a�]�Z�V�N�E�A�A�A�A�A�A�g�f�f�g�p�s���������������������s�g�g�g�6�)����$�6�O�[�e�h�l�n�k�h�[�O�J�B�6ā�|āąčĎĚĦħĩħĦĚčāāāāāā���y�l�]�f�l�����Ľн��������н���!���!�.�:�@�G�S�[�S�G�:�.�!�!�!�!�!�!��������ùùù��������������������������ÓÉÇÀÇÓàèéàÓÓÓÓÓÓÓÓÓÓ�x�w�m�x���������������������x�x�x�x�x�x���������"�$�*�&�$�������������������������������	����	��������������������ĿľĺĻĻĿ������������������������EuEkEiEbE_EiEuEuEvE�E�E�E�E�E�E�EuEuEuEuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������'�(�(�'�$��������'�&�+�4�5�@�D�M�Y�f�k�f�f�^�Y�P�M�@�4�'D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��ù������������ùĹϹܹܹܹܹҹϹùùù� > . | � 9 0 ! R P u  3 L 9 = h " J M R . B _ 9 Y ( : < < * 0 Q ! Q - B 8 $ F 8 @ i U 9 @ 0 8 : X Y T & A * i T 1 G 7 � Y  X J c ^ -    R  u  �  �  �  �  V    ]  o    9  �  �  o  �  �  �  S  9  �  �  w  �  �  I  :  �  �  0  �    U  ,  �  �  w  �  5  �  �  �  6  o    �  �  *  %  �  �  �  �    �  �  k  a  �  �  M  �  �  �  9    �  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �        $  *  .  1  .  '    �  �  �  �  n  S  C  B  G  �  �  �  �  �  �  |  w  p  g  ]  T  B  *    �  �  �  q  A  �  �  �  �  �  �  �  �  m  U  0  �  �  _  6  !  "  (  0  8  �  �  �  �  �  �  �  n  K  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  l  ^  O  =  )    �  �  �  �  G     �  /  <  P  ]  d  f  c  V  >  !    �  �  �  a    �  s  7      %  ?  M  U  Z  [  X  S  A  $  �  �  z  .  �  �  4   �   y  j  �  �  z  O  '     �  �  �  _  3    �  �  {  I    �  �      /  9  9  .      �  �  �  g  :  
  �  �  �  �  a  z  ;  7  2  .  *  &  "          �  �  �  �  �  �  �  �  �  b  w  y  n  _  M  :  %    �  �  �  �  �  �  f  2  �  _   �  �  �  �  s  g  X  I  5      �  �  �  �  o  B    �  �  O  |  r  ^  C  $    �  �  �  f  :  
  �  �  ~  :  �  �  -  �  z  �  �  �  �  �  �  �  �  �  �  �  u  6  �  x  �  #  H  �  l  �  �  �  u  J    �  �  Q  �  �  :  �  �    	�  �  s  
  V  U  T  S  O  J  E  <  1  %      �  �  �  �  �  �  �  q  �  �  �      !  %  $         �  �  �  �  X    �  *   �  �  �  �  �  �  u  a  F  -      �  �  �  �  �  �  b  )  �  @  ,         �  �      7  ;  ;  :  6    �  �  �  �  p  F  ;  1  &        �  �  �  �  �  �  �  �  �  �  w  l  b  =  6  ,  #      �  �  �  �  �  {  Q  #  �  �  r       J  >  H  9  !    �  �  �  �  b  9    �  �  �  �  �  �  �  }  �  �  	  �  �  �  �  �  �  �  �  h  1  �  �  w    �  �   �  �  �  �  �  �  �  �  �  �  r  M  #  �  �  �  q  9     �   �  �  �  �  �  s  M  2  I  K  0  	  �  �  s  <    �  �  �  �  �  w  j  ^  P  A  0  #      �  �  �  �  �  V    �  �  8  ~  {  x  u  r  o  l  i  f  c  `  [  W  R  N  &  �  �  �  Z      ,  =  G  L  G  >  3  $      �  �  �  �  �  �  �  �  {  p  e  Z  L  >  .    	  �  �  �  �  �  �  �  ~  �  �  �  �  �         (  -  *      �  �  �  �  i  (  �  �    a  �    '  E  H  @  1    �  �  �  �  ^  )  �  �  .  �  7  �    @  P  W  U  O  C  3       �  �  �  m  6  �  �    x  )  u  �  �  �  �  x  e  M  3    �  �  �  �  {  ^  8    �  �  �  �  �  �  �  �  �  �  v  T  +  �  �  �  S    �  �  M    V  G  4      �  �  �  �  l  J  &    �  �  �    c  H  '  7  C  K  L  F  5    �  �  �  7  �  �    �    D  [  i  W  �  �  �  �  �  �  �  x  p  g  _  V  N  F  A  ;  5  0  *  %  �  �  �  �  k  Q  6    �  �  �  �    ]  5    �  �    *  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  z  w  �  �  �  �  �  �  �  �  �  v  S  (  �  �    1  �  �    s  �  �  �  �  �  �  o  O  /    �  �  �  �  �  l  k  H    �  �  �  �  �  �  �  �  �  �  �  �  {  p  b  S  E  .    �  �  h  _  U  K  E  D  C  A  ;  ,    �  �  |  '  �  Y  �  Q   �  �  �  �  �  �  �  �  �  t  \  A  $  �  �  h    �  �  N  �  	�  	�  	�  	�  	s  	K  	   �  �  \    �  x  .  �  �  9  �  �  �  L  M  N  O  P  P  K  F  @  ;  2  &         �   �   �   �   �  ^  P  A  3  ,  +  '  !      �  �  �  �  �  j  J  +  ,  9  �  �  �  �  �  �  y  _  K  7      �  �  �  p  >     �   x  b  \  U  Q  S  W  V  M  ;  '    �  �  �  m  +  �  �  &  �  z  p  g  ^  U  M  E  =  2  #      �  �  �  �  �  �  _  5        �  �  �  �  �  �  �  |  _  @    �  �  �  L     �  
]  
�  
�  
�      
  
�  
�  
U  	�  	c  �  <  �  �  x  <  �  Y  �  �  �  �  �  m  E    �  �  �  �  v  ]  C  *    �  �  �  2      �    �  �  �  d  ;    �  �  �  �  �  �  �  P  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  a  I  1    3  H  ?  5  )    �  �  �  �  �  �  �  �  �  �  �  �    6  -  %          	      �  �  �  �  �  �  �  �  �  �  n  Z  G  3      �  �  �  �  �  i  N  2    �  �  �  �  q  �  �  �  �  �  v  c  O  7       �  �  �  p  F        �  �  F  4  "      �  �  �  �  �  �  �  v  U  4    �  �  �  �  �  �    l  b  k  �  �  �  �  v  d  F  !  �  �  �  5  �  �  	n  	�  
#  
D  
[  
U  
E  
.  
  	�  	�  	|  	  �  �  F  �  �  �          '  1  4  3  .  "    �  �  �  O  �  �  *  �  n    �  �  �  c  =    �  �  �  j  ;  
  �  �  �  g  �  l   �   7  �  �  ~  k  U  >  %  
  �  �  �  �  o  P  0    �  �  x  f  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  n  �  �  ~  F    �  �  �  �  z  Y  6    �  �  z  *  �  �  �
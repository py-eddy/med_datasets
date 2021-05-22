CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�KƧ     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NF�   max       P��y     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �   max       <�C�     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @F��Q�     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(��    max       @vx�\)     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @��`         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �7K�   max       <#�
     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�
�   max       B4��     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��
   max       B4�9     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =���   max       C��[     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��   max       C��C     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          [     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NF�   max       PJ	e     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�&���   max       ?�^5?|�     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �   max       <�C�     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @F�          �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(��    max       @vxQ��     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�#�         (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e+�   max       ?�^5?|�        `�         $      
      
                            *         *   )   	         	                      '                     T   
      
   A   I               	         
   
   
                  E      	         &      +       *      [         5   P�IN��O�ۚO�ǑN��O�xN�_�O�~IO�QN���N��P(QRN��P5�N��PP��N��P�1O�$MO��;N��N�^N���N���O=ȆNF�O��KN.�Oֆ�O E&P��N0�%N"CNf�N�POw�N�ȂP��N�"NW�N�(�O�qP��yO~*#Oh��N7�KN�VWO��NgX�N���N���OZ�dO�O@i�N���O��N���O�(O��]OZzZN���O��N�M_O��O?�1P��Ow�NO�32Nq�O�QN8C�N�qOP�N�G<�C�<D��;��
;o:�o:�o:�o�D����o�ě���`B�t��T���T���T����o��o��o��C���C���t����㼛�㼴9X�ě��ě����ͼ��ͼ�/��`B��`B�����o�o�o�\)�\)�t��t���P��w��w�#�
�',1�,1�0 Ž0 Ž49X�49X�49X�49X�<j�<j�@��@��T���Y��aG��}�}�}󶽁%��%��7L��\)������w��E���E���j���������)5CJHB5����ENV[gkhg^[WNEEEEEEEE7=HNTX`aimgefaTH<347�������������������� )+0*)��     
#$./320/$"
� ��������������������z~��������������zusz *46@CEEDDC>6*����������������������������������������)19B[t�����hB)ggt{����������~ttggg�����#&���������fhju���������ujhffff��03<<3)�������')+105BDHIJB@5))''''����������������������������������������)EO[ehpth[NB6))56BN[`glgf[NB;5))))!#$/9<@<<5/###""!!!!��������������������������������������������������������������������������OTamz������|zmaZTMJO��

�����������Zamz��������zmhc`[XZ������������������������
��������;;HOLIH;65;;;;;;;;;;��������������������
#/83/#
)**)��������������������zz�������������|zzz���������������������������������������������������������
"
�������O[hnrrh[OB)	)O���23.'��������������������������������������������������������~y|���������	  ��hmz���������zqmmffhh#/373/#"Q[gt|���tg[PQQQQQQQQyz|������������|zzyy��������������������IOTUbbelonkcbUTJIBAI������������������������� ��������������������������?IUbffbbUIGC????????���������������������5B[m{tgN5)��������������������������������������������#)04<AHIQI<30-#Wbhn{����}{nb`[WWWWtx������������}vrqqt|��������������xtuw|%3:Nt�������gXIB5)#%��������$������Sanz���������xaUNKMSHMLIH<7536<HHHHHHHHH����
#/<JKHE</
���xz������zuxxxxxxxxxx��������������������aknrohaUH<3/../3<HUaTU_ahhba`UTRTTTTTTTT���������g�O�K�P�L�Z�g�s�x���������������	����������	������	�	�	�	�	�	�	�	����ƮƧƝƚƎƊƎƚ�������������������������������
��#�/�=�=�E�<�/��������B�;�6�5�3�/�5�B�N�P�[�[�f�e�[�N�B�B�B�BE�E�E�E�E�E�E�E�E�FFFF$F$FFFE�E�E澥���������������������ʾʾ̾ʾž����������������������)�B�O�P�J�M�I�B�(���ܿ.�"��	����������	��"�'�.�3�8�8�4�.���ݽսڽݽ�����������������!��!�-�5�:�;�F�M�F�:�-�!�!�!�!�!�!�!�!���������m�g�r�x�t�y�����ĿԿ߿ۿݿѿ����m�k�e�`�^�\�`�j�m�q�y�|�������y�y�n�m�m�f�]�_�l�o�x�}�����;׾������ʾ�����f�G�@�;�5�2�7�;�A�G�K�T�]�W�Y�T�P�G�G�G�G�T�?�%��"�;�T�a�z�����������������z�a�T����������������������������������������Ŀ����n�m�y�������Ŀѿ���������������������������0�=�C�C�A�=�0�$�����S�L�L�F�C�B�F�o�x�����������������x�_�S�����������������������������������������U�J�H�A�H�L�U�a�b�n�y�z�{�z�n�a�U�U�U�U������)�6�>�B�M�N�B�?�6�)��������������ʼмּ������ּʼ������������3�(�"���
���(�4�A�M�Z�e�f�]�Z�M�A�3�a�a�U�R�O�U�a�d�g�b�a�a�a�a�a�a�a�a�a�a�����������	��"�/�H�M�a�d�e�W�;�"��	����������������������������������������ĳĨğĥ���������
�!��������������ĳ��������������(�5�A�C�A�5�-�(����Z�P�R�R�Z�g�s�����������������������g�Z�A�A�<�A�N�Z�^�d�Z�N�A�A�A�A�A�A�A�A�A�A�l�g�b�l�x��������x�l�l�l�l�l�l�l�l�l�l����žŹŸŶŹ���������������������������z�o�u�y�z�����������z�z�z�z�z�z�z�z�z�z�Y�Q�M�@�5�5�>�B�M�Y�]�f�u���������f�Y�����������������!�������'�����&�4�@�f�r������������r�Y�4�'�t�i�g�e�c�g�t�t�t�t�t�t�t�û»��ûƻлܻ����ܻлǻûûûûûû������������!�,�-�0�4�4�-�!����y���������������������������s�g�[�P�M�y���������S�l�����������ݽ�׽����S�:��������"�:�F�S�_�d�h�f�_�S�F�:�-���������v���������ùϹ�������ܹϹù���Ƴ����������������ƳƧƥƧƳƳƳƳƳƳƳ����ùñìëìù�����������������������ſĿ����Ŀǿѿؿݿ������������ݿѿĿ�������$�0�3�1�0�&�$��������������������	�	����	�����������������'�������'�'�3�=�@�L�W�L�@�<�3�'�'�Y�X�e�q�o�{���������ɺֺͺ������~�r�e�Y�����������������ûл�������ܻۻлû��`�T�H�G�;�7�;�G�T�m�����������������m�`���	�����ؾ۾����	���"�.�0�.�"�����������������������
����������޻-�&�&�)�-�:�F�I�R�K�F�:�-�-�-�-�-�-�-�-ƳƨƧƣƥƧƲƳ����������������������Ƴ�J�J�S�[ĀčĞĦĿ������ĿļĪĚā�h�[�J�0�'�"�0�=�I�R�V�b�o�{ǀǈǑǓǈ�{�V�H�0�t�k�h�g�h�s�t�uāčėĚĦĚčā�t�t�t�t�������������������������ĽĽĽ����������Ľ��Ľʽͽнڽݽ�������������ݽнĽ�ŝŔŇ�z�u�{Ŋŭ������������������Źŭŝ����������*�6�C�K�Q�O�J�C�6�*���#������½»¾����������#�<�P�]�R�<�#��ּ����¼ʼϼּܼ��������������T�G�A�?�B�K�a�m�z��������������z�m�a�TD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�EEED�D�D�EEE0ECE\EbE\EWESEQELEAE*E���������������ɾǾ����������������������������������������������������������忟�������Ŀѿ׿޿ݿؿѿɿĿ�������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� j m q w G > . g H . O S P 4 S 4 V > " . Z U e K ; m K I Q g Z I h ~ d 2 K 4 4 Y 3 Q W F h g S + G b [ \ F f � T = I � n N - 9 7 ' l Z : K W F   I    u  9  �  �  d  �    ]  �  8  9  �  �    �  �  �  �  -  C  �  �  �  �  T  �  S  %  L  �  E    �  m  �    �  �  �    k  c  !  .  z  �     �  �  �  0  l    Y  =  �  C  n    �  %     �  �  )    �  �  �  Y  �  �  O%@  <#�
����C��t���/�t������49X�e`B�#�
��h��t��49X����m�h��9X�'q���ixռ�/���o����P��`B�Y��+�q���'�+�o�t���w��P�e`B�D����F�<j��w�<j����S���\)��hs�49X�Y��P�`�aG��P�`�]/�Y��]/�}�P�`����]/��+���ٽ����hs��E�������������G������xս���7Kǽ�j��
=�!����B�IB�5A�I�B{~B�pBvAB4��B!`B/�(B!gWB-G�B�!B
BB#+EB36B��Bd�B+4�B�BҐBrB@9B�TB!�B��B�A��B�WB +B��B� A�
�B ��Bv�B��B \lB+B<�B��B*2hB#�IB�FB_AB��B�B
b�BŜA��SB�dB	?�B��B�0B't&B�VB6rBj�B'OB��B	iB=�B�B%��B(?pBB
�JBćB-�kBd�B�=B�B��B�=B�BoB��B�aA��VB@zB��BA;B4�9B �.B0!:B!@TB-@�BPAB
CHB$9�B3=�B��B}�B+;B)�B9�B#RB?�B��B!�hB�B>�A��FBтA��*B�(B;�A��
B �UB�OBS�B C�B@B;}B/�B*B�B#��BMqBw2B�
B9�B
iB�A�ܴB�RB	s�BȶB
JB'AeB�:B|�B%B&�B�JB"2B��B��B%��B(1�B?�B
��BH�B-�zBB�B��B�}B��B��BUIB<OA��A�7B�A��oA�&C��[AL��AԥwA\�KA-�@xB�At�*Ak'�AK�AeA�&[A��AyB	,5@��2Aq7FA��A��A s�A:ARA�>9A�l_A��A���A�R�A�r;A��`@��A�K�A��@�V�@�}�@�I�A��x@��
@h�A��dAI�@y2u=���BѕA�cqA|�B	�AZi?���@�@�#Ak��AZPA�2b@z2�B�A�`CB�Aݮ=A"�A+�A��kA�X-A�^�A��A�@C��;C��oAN�A�'WAw�uC�sA���A�'�BĤA�P�A���C��CAL�?Aԅ~A[0�A-�@z�	AuaAlAJAd�
A��A��_Az�TB	�L@�mhAp�HAń-Aր�AxA;ƝA��A��A��A�XiA��A���A���@�NKA��A��y@݀�@���@ڻ�A��@�!4@f�/A��A��@���=��B�A�x�A}�B	�AZ��?�z\@!m@��9Aj��AWg�A���@taB��A���B��Aݗ�A!_�A+�NA���B hwA���A�zA�yxC��C��AN�'A�d�Aw/�C��         $                     	                   +         +   )   
         	            	   !      '                     U         
   A   I               	            
                     E      
         '      ,       *      [         5      -      !               #            +      )      -      -   !   #                     !      #      )                     )            &   9                                                +               !      -      !      #               -                     #            +      )      -      -                                 !      )                                    +                                                '                     -      !                  P�IN��O2��O�N��N�:�N¢mO�~IO�QN���N��P(QRN��P5�N��PJ	eN��P�1O��O�$oN��ND��NE�N���O6�NF�O/$�N.�O���O E&P��N0�%N"CNf�N�POS�N�ıO���N�"NW�N�(�Oa�P80�O~*#Oh��N7�KN�+VO��NgX�N���N]�O8�N���O@i�N���O��N���N��vO�4OOD��N���N�OcN�M_Oj�NO?�1P��Ow�NO�32Nq�OUq�N8C�N���OP�N�G  �  �  Y  t  i  �  c  n  �  {  v  �  �    g         �  �  8    �      �    �  �  K  c  �  '    �  �  �  	�      �  �  `  �  �  �  l  :  �  �  �  m  /    �  �  V  �  �  �    �  �  �  �    �  f  �    q  �  �  m<�C�<D���ě��D��:�o��o%@  �D����o�ě���`B�t��T���T���T����C���o��o��j����t���1�ě���9X��/�ě��t����ͼ���`B��`B�����o�o�#�
���y�#�t��t���P����P�`�#�
�',1�0 Ž0 Ž0 Ž49X�8Q�8Q�8Q�<j�<j�@��@��Y��y�#�e`B�}󶽅��}󶽗�P��%��7L��\)������w��/��E�������������)5CJHB5����ENV[gkhg^[WNEEEEEEEE9<CHLTabcbba_\THE<99�������������������� )+0*)��    	
#(,/00/,#

		��������������������z~��������������zusz *46@CEEDDC>6*����������������������������������������)19B[t�����hB)ggt{����������~ttggg�����#&���������fhju���������ujhffff���.2;;2)������')+105BDHIJB@5))''''����������������������������������������)6BOV]]\OEB6))56BN[`glgf[NB;5))))"#)/4<=<92/'##""""""��������������������������������������������������������������������������_ajmz�������zmmea^Z_��

�����������]amz���������zlfc^[]������������������������
��������;;HOLIH;65;;;;;;;;;;��������������������
#/83/#
)**)��������������������~�����������~~~~~~~~�����������������������������������������������������������
"
�������)6<GQX[ROB6)����.,'�����������������������������������������������������~y|���������� hmz���������zqmmffhh#/373/#"Q[gt|���tg[PQQQQQQQQzz~�����������{zzzzz��������������������CIKRUbbillkibUULIDCC������������������������� ��������������������������?IUbffbbUIGC????????���������������������5B[_hg[N5)�������������������������������������������� #-01<?FHI<0#     Wbhn{����}{nb`[WWWWz���������������{xxz|��������������xtuw|%3:Nt�������gXIB5)#%��������$������Sanz���������xaUNKMSHMLIH<7536<HHHHHHHHH��
$*010#
������xz������zuxxxxxxxxxx��������������������aknrohaUH<3/../3<HUaTU_ahhba`UTRTTTTTTTT���������g�O�K�P�L�Z�g�s�x���������������	����������	������	�	�	�	�	�	�	�	������ƷƳƩƮƳ���������������������������������������������
��"��
���������B�;�6�5�3�/�5�B�N�P�[�[�f�e�[�N�B�B�B�BE�E�E�E�E�E�E�E�FFFF FFFE�E�E�E�E澥���������������������Ⱦʾ˾ʾþ����������������������)�B�O�P�J�M�I�B�(���ܿ.�"��	����������	��"�'�.�3�8�8�4�.���ݽսڽݽ�����������������!��!�-�5�:�;�F�M�F�:�-�!�!�!�!�!�!�!�!���������m�g�r�x�t�y�����ĿԿ߿ۿݿѿ����m�k�e�`�^�\�`�j�m�q�y�|�������y�y�n�m�m�f�]�_�l�o�x�}�����;׾������ʾ�����f�G�@�;�5�2�7�;�A�G�K�T�]�W�Y�T�P�G�G�G�G�a�T�@�&��$�;�T�a�z�����������������z�a����������������������������������������Ŀ����n�m�y�������Ŀѿ������������ �������������$�0�:�@�A�>�9�0�$���l�_�W�V�R�O�N�S�l�x�����������������x�l�����������������������������������������U�O�H�D�H�R�U�Z�a�n�s�n�a�_�U�U�U�U�U�U�����)�6�A�B�E�D�B�6�2�)���������������ʼмּ������ּʼ������������4�,�(�����(�4�A�M�Z�_�a�Z�U�M�A�4�4�a�a�U�R�O�U�a�d�g�b�a�a�a�a�a�a�a�a�a�a�	��	�
����"�/�;�H�I�V�U�T�H�;�/��	��������������������������������������ĳĮĤĩĶ��������� �
�������������ĳ��������������(�5�A�C�A�5�-�(����Z�P�R�R�Z�g�s�����������������������g�Z�A�A�<�A�N�Z�^�d�Z�N�A�A�A�A�A�A�A�A�A�A�l�g�b�l�x��������x�l�l�l�l�l�l�l�l�l�l����žŹŸŶŹ���������������������������z�o�u�y�z�����������z�z�z�z�z�z�z�z�z�z�Y�W�N�M�H�B�H�M�Y�f�k�r�w�����r�f�a�Y�����������������������f�Y�M�9�4�-�*�,�4�@�Y�f�r����������r�f�t�i�g�e�c�g�t�t�t�t�t�t�t�û»��ûƻлܻ����ܻлǻûûûûûû������������!�,�-�0�4�4�-�!����s�m�g�c�l�s���������������������������s����� ���!�`�y���������������q�S��������"�:�F�S�_�d�h�f�_�S�F�:�-���������v���������ùϹ�������ܹϹù���Ƴ����������������ƳƧƥƧƳƳƳƳƳƳƳ����ùòùü���������������������������ſĿ����Ŀǿѿؿݿ������������ݿѿĿ�������$�0�3�1�0�&�$��������������������	�	����	�����������������'�%������'�(�3�<�@�H�@�3�,�'�'�'�'�r�f�s�q�~�������������ɺκĺ��������~�r�û��������������ûлܻ����ܻٻлûÿ`�T�H�G�;�7�;�G�T�m�����������������m�`���	�����ؾ۾����	���"�.�0�.�"�����������������������
����������޻-�&�&�)�-�:�F�I�R�K�F�:�-�-�-�-�-�-�-�-ƳƫƧƤƦƧƳƴ����������������������Ƴ�N�M�X�eăĐĥĦĳĵĿĽļĸħĚā�h�[�N�0�+�'�0�=�I�T�b�o�{�ǈǐǒǈ�{�V�I�A�0�t�k�h�g�h�s�t�uāčėĚĦĚčā�t�t�t�t�����������������������½����������������Ľ��Ľʽͽнڽݽ�������������ݽнĽ�ŠŕŔŇņŃŔŠŭŹ��������������ŹŭŠ����������*�6�C�K�Q�O�J�C�6�*���#������½»¾����������#�<�P�]�R�<�#��ּ����¼ʼϼּܼ��������������T�G�A�?�B�K�a�m�z��������������z�m�a�TD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�EED�D�EEEE*E7ECELELEGECE;E7E*EEE���������������ɾǾ�����������������������������������������������������������忟�������Ŀѿ׿޿ݿؿѿɿĿ�������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� j m T ^ G @ * g H . O S P 4 S 4 V >  ' Z : r K @ m > I L g Z I h ~ d 7 ; 4 4 Y 3 N / F h g E + G b K O 6 f � T = B  h N  9 + ' l Z : K + F   I    u  �  h  �    �    ]  �  8  9  �  �    |  �  �  l    C  N  �  �  3  T  �  S  �  L  �  E    �  m  =  �  ~  �  �    �  I  !  .  z  �     �  �  ~  �      Y  =  �    �  �  �  �     �  �  )    �  �  �  Y  �  �  O  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �  |  l  x  �  �  �  ~  u  h  W  B  *    �  �    J    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    <  O  Y  Q  4     �  �  >  �  �  U  	  �  n  F  Y  X  F  ;  s  t  `  D    �  �  d    �  �  b  =    �  k  i  _  S  7      �  �  �  �  �  t  \  D  $     �  L  �  k  �  �  �  �  �  �  �  �  ~  O  "  �  �  �  B    �  �  �  F  Z  _  b  `  ]  W  O  E  9  +      �  �  �  �  S     �   �  n  \  :       �  �  �  g  $  �  �  1  �  �  �  �  F  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  k  ^  Z  Y  X  {  s  j  `  U  I  ;  -      �  �  �  �  {  R  %   �   �   r  v  o  i  b  [  U  N  H  B  <  6  0  *  #         �   �   �  �  �  �  �  �  �  �    r  a  M  4    �  �  �  y  C  .    �  �  �                    
      �  �  �  �  �      �  �  �  �  �  {  L    �  �      �  �  �  L  �  D  g  E  $    �  �  �  �  t  Z  C  0    	   �   �   �   �   �   �            �  �  �  �  �  �  �  �  �  �  J  �  �  
  x  Y        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  c      �  �  �  �  �  �  �  �  �  �  �  �  �  f  =    �   �  �  �  �  �  �  �  �  �    X  .  �  �  n    �    �  �    �  �  �  �  �  �  �  �  �  �  �  Y    �  ~    �  �  y  _  8  1  *  "        �  �  �  �  �  �  i  <    �  �  I   �  �  �        	  �  �  �  �  �  �  j  J  *  	  �  �  �  z  7  O  e  z  �  �  �  �    P  �    �    �  K  	  	�  
{  =      �  �  �  �  �  �  �  �  �  {  i  7    �  �  �  k  D  �              	      �  �  �  �  �  y  _  B  !  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  L  1    �  �  �  �  �  �          �  �  �  �  `     �  �  7  �  �  �  �  �  �  �  �  �  �  �  w  e  R  =  '    �  �  X  �  �  �  �  �  �  �  �  �  n  I    �  �  [  �  �    �  I    K  @  /      �  �  �  �  p  \  �  �  u  I    �  �  �  G  c  _  N  4    �  �  �  I    �  w  4  �  �  �  �  |  �  a  �  �  �  �  }  {  y  w  u  r  k  _  S  G  ;  /  #       �  '  &  &  %  $  #  "  !          �  �  �  �  �  �  �  �    �  �  �  y  R  0    �  �  �  v  Q  ,  
  �  �  �  �  {  �  �  �  �  �  �    {  z  {  |  ~  w  j  ]  P  ?  ,      o  z  �  �  �  �  �  �  �  �  {  _  ;    �  �  l    �  Z  �  �  �  �  �  �  �  �  �  �  `  9    �  �  ~  D  �  �   �  �  	  	H  	t  	�  	�  	�  	�  	�  	X  	"  �  �  E  �  .  q  �  �  �    �  �  �  �  �  �  �  q  I    �  �  �  e  .  �  �  �  �       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  h  �  �  �  �  �  r  b  S  C  3  #    �  �  �  �  �  �  ~  g  V  Z  Z  f  m  r  x  �  �  s  O  !    �  W  �  4  ?  �  {  �  �  K  _  \  Q  @     �  �  b    �  )  �  �  e  �  
  
  �  �  �  �  �  �  h  G    �  �  r  ,  �      �  K  �    �  k  N  5  !    )    �  �  �  �  _  .  �  �  �  H    [  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  d  i  e  L  0    �  �  �  }  E    �  �  e  6    �  �  �  :  3  +  $          �  �  �  �  �  �  �  m  R  6    �  �  �  �  �  �  �  �  l  R  4    �  �  �  |  R  &  �  �  �  �  �  �  �  y  n  a  T  F  5  #    �  �  �  �  �  �  �  ~  I  o  �  �  �  s  Z  @  %  	  �  �  �  �  �  �  �  �  �  r  Y  b  l  c  W  J  <  3  .  #    �  �  �  �  �  f  E     �  +  -  .  *  %          
  �  �  �  �  �  �  �  �  a  >           �  �  �  �  �  �  �  �  \    �  �  B  
  �  �  �  �  �  �  �  �  �  �  �  �  �  r  f  ]  S  I  8  %       �  �  �  s  r  Z  ;      �  �  �  ]  )  �  �  �  �  z    V  F  7  &      �  �  �  �  �  �  u  d  z  �    "  !     �  �  �  �  �  �  �  b  ?    �  �  �  u  E    �  �  [    q  v  �  �  �  z  d  z  C  �  �    
�  
   	�  �  �  g  G  �  �  �  �  �  �  �  �  p  R  *  �  �  �  H    �  `  �  u   �    s  g  X  I  5      �  �  �  �  �  �  �  w  X  7    �  s    �  �  y  g  R  8    �  �  �  F  �  p  �  Y  �    2  �  �  |  d  I  ,  
  �  �  �  �  �  n  ]  M  <  %    �  �  �    =  _  x  �  �  ~  m  T  2     �  l  �  h  �  �      �  �  �  ~  o  ^  J  7  %      �  �  �  �  �  �  T  �  e      �  �  �  [     �  �  �  �  �  �  �  �  v  �    N  �  �  �  �  �  s  H    �  �  �  �  z  D    �  d  �  �  �  k  f  U  C  1  !    �  �  �  C  �  �  F  �  �    �    O  �  �  h     �  �  r  *  �  �  �  K    �  x  ,  �  �  <  �  �  �  j  �      �  �  �  7  �  w  �  .  R  D  
.  �  �    �  q  g  ^  T  J  @  7  +        �  �  �  �  �  �  �  }  k  �  �  �  �  �  �  u  _  A    �  �  r  1  �  �  /  �  p   �  �  �  e    �  `    
�  
f  
  	�  	3  �  )  �  �  '  o  �  �  m  @    �  �  �  �  ]  7    �  �  u  7     �  Y    �  s
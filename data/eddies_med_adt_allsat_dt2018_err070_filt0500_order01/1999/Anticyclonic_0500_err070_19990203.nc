CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?��
=p��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MѠ�   max       P�W�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       =�
=       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @Fb�\(��     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ٙ����    max       @vy�Q�     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @P�           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�A�           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >-V       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�8   max       B2�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B2�       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =^,�   max       C���       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =Co2   max       C��       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MѠ�   max       PI��       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�,<�쿲   max       ?��S���       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       =�
=       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @Fb�\(��     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �θQ�     max       @vy�Q�     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P�           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���           V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F
   max         F
       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-�qv   max       ?��S���     P  X�                  	      �                        J   +         4      5      I                        8      �      :   
   N                     #   g                  !         	               
      *      
   )O��Ni��MѠ�N$��NS�N���N���Pv�3N��_N��CNQ(�O;&CN�d{N�ffN~�P�W�O��tN^'�N
P<�7OW>+O�N�N4<P\9�N��N�O��O�JN�Y�N�.�O!F�PL�ONO_jP	a�N��PBr9OKP�P�N+O\(4N8�OAOe�OwO�@�PDvRN���N���N�C�N{��N�1*O4�GN��N)oN�N���N�K�N�~�N���N.	�Nl^OP݈N�-�N6Q�N�1��+�o��o��o�u�T���D���#�
��`B��o;��
;��
;��
;�`B;�`B<t�<49X<D��<D��<e`B<���<��
<��
<�9X<�j<�j<�j<�j<�j<ě�<ě�<���<���<�/<�/<�`B<�h<��<��<��=\)=\)=�P=�P=�P=#�
=#�
=#�
=#�
=0 �=<j=H�9=H�9=T��=y�#=y�#=y�#=�%=�C�=�t�=�t�=�t�=��=���=�
=cagt������������ztgc(!!)168>>;6)((((((((YVY[\gikg[YYYYYYYYYY��������������������aUKH=<::<HSUaaaaaaaa����������������������{}����������������	BNg|������tgPB	"$/4;ECB;/"eglt�������tngeeeeee��"��������WPQRX[bhtw�����wth[W2-46CO\ab\[QODCB7622VPOQ[htwz{xtrh`[VVVVv�������������vvvvvv������:R[mfB)������������������������������������")6?BFCB96+)""""""""�����
/HU_cnnaH<����#/7<HPRSRNH</#�������
"!
����#%).6;BCBA=62)######;6>Ngt�����������[N;335;HT`\UTH;33333333 �
##$&&#
    TUW\anyz|��zna[VUTT��������������������??BO[bdg[OGB????????��������������������ahjt����������tpnjhanpkko�������������vnRP\hu}wuh\RRRRRRRRRR����
"0EG>1#
���������� 
�������{}�����������������{rr{����������������r������&-+/)�����$%),+)!���������������
"
���������|{���������������||"/;HTWYVPHC;3/"#/<EHMTXUSH<6//(#������������������ppz�������������zp./7<HRUW^UTHA<4/....�������������������� 
#(.+(#
      ��������������������e`ahmt�����vtheeeeee����thg_a__adhtx����776*$*-777777777UUUHGDHHMU[[UUUUUUUU]XYamxz|}|zmfa]]]]]]������

����������
"#)*$#
()+566976/)0066BBO[db[OMB:60000������������������������������������������)6BHNKB96)�#!!"""#//3;<<<:5/###wnpyz|����zwwwwwwww���������������������
�������
� ��������������������
�����������������������������������������������������������������������������������������������������E\EXEPEMEPEPE\EiEoEuEiEdE\E\E\E\E\E\E\E\���������������������{�|���������������������������ùƹϹԹѹϹù�����������������)�B�T�y�v�h�O�6�� ����üþ���������a�b�j�m�v�z���z�m�a�]�[�\�Z�a�a�a�a�a�a�T�V�a�e�f�a�^�T�H�C�@�F�H�P�T�T�T�T�T�Tù��������������ýùìîùùùùùùùù�4�A�M�Z�f�k�s�{�v�s�f�M�A�4�1�(�&�'�.�4�G�T�`�b�i�k�d�`�T�H�G�;�6�.�&�.�;�<�G�G�ʾ׾����������׾ʾ����������ʾʾʾʿ	���"�.�3�.�+�"���	���	�	�	�	�	�	����������ݿϿ��m�M�;�.����"�.�^������(�4�D�J�C�4�(����ݽ����}�{��������5�A�N�N�T�S�N�A�<�5�*�0�5�5�5�5�5�5�5�5ÓÝÜàãàÓÑÇÄÇÉÓÓÓÓÓÓÓÓ����/�@�G�B�-�)�"��	�������������������s�����������������������������p�c�`�g�s�)�5�B�[�`�e�d�[�N�C�5�)�����������)�ûлֻܻ����߻ܻлǻû��ûûûûû��5�N�g�}���������j�U�5�������	���5���(�0�*�0�(������������������������ɾ���������������������������������������������������~�x�x����������y�u�u�s�n�k�Z�M�A�4�2�4�7�A�M�Z�f�n�r�y�z�����������z�m�i�i�m�r�z�z�z�z�z�z�z�zčďĚĦĨĲĳĳĳĦĚčćĂăččččč�'�1�4�:�9�4��������ܻѻ������'�M�f�����������������f�M������4�M�׾�����׾־оо׾׾׾׾׾׾׾׾׾׺ֺ��7�:�S�:�-�����ɺ������������ֻ������������������x�l�_�S�N�S�_�g�x�������������������������g�R�F�@�=�B�N�Z�g���������!�-�2�:�I�F�:�!������������I�U�[�nŇŐŖőŇ�n�<������������1�@�I�/�<�H�J�H�<�9�/�#��#�(�/�/�/�/�/�/�/�/�����"�.�;�G�Q�G�>�;�,�	����۾ھ����a�g�n�o�s�n�a�^�U�M�U�X�a�a�a�a�a�a�a�aE�E�FFFF F$F%F$FFE�E�E�E�E�E�E�E�E��
��#�*�-�1�/�'�#��
�����������������
��� �!�"����������������������
��������ܹ���������ܹ�������������Óàñ��������ìÓ�z�n�c�Z�O�W�sÂÈÄÓ�z�{ÇËÏÓÖÓÇ�|�z�r�n�j�n�x�z�z�z�zìù��������üùìàÜàâèìììììì������	�������ڼڼ�������ｅ���������������y�u�w�u�y�~�������������@�L�Y�_�e�q�k�e�Y�L�F�@�?�>�@�@�@�@�@�@�F�;�-�+�'�,�-�:�F�S�_�l�x�|�}�x�n�_�S�F�x�l�l�i�i�l�y�}�������x�x�x�x�x�x�x�x�x�5�5�5�?�B�N�[�\�[�Y�N�B�5�5�5�5�5�5�5�5ŭŹ����������ŹŭŠŜřŠŦŭŭŭŭŭŭD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Dƾ������žƾ����������������������������������������������������~�|�~�������������Ŀѿܿݿ�����ݿڿҿѿɿĿ��ĿĿĿ��)�6�8�6�0�6�@�6�)�����$�)�)�)�)�)�)������������������������������������������������������������������}�z�}����������"�.�;�G�T�Z�T�T�G�;�6�.�(�"������M�Y�f�r�t�r�j�f�Y�S�M�K�M�M�M�M�M�M�M�MDoD{D�D�D�D�D�D�D�D{DoDkDhDnDoDoDoDoDoDo o 4 L G ` Z r X { ; 6 % 3 $ ] \ � 8 ] ` S $ p G 5 , + V 1 < m & a 7 { 8 � I k T b 6 + N g > b E 9 q 5 6 A I G b / - w Q R P k $ %    _  w  
  [  �    �  @  �  �  q  �      �    0  n  _  �  �  �  �  �  �  �  %  |  �    �  Z  �  �  K  `  �  �  :    y  5  �  L    c  �  �  �  �  �  �  =  L  �  �  �  �  �  ^  �  �  �  Q  ۼ����`B�D���49X��`B��o;o>t��D��:�o<�9X<�9X<D��<�C�<D��=��=Y�<���<u=��=t�=��<���=\<�/=�P=�w=t�=��=,1='�=��
<�`B>-V=D��=�{=�w=�/=C�=0 �=,1=P�`=��=m�h=�hs>n�=P�`=Y�=D��=H�9=�o=��T=T��=q��=�\)=��-=�O�=��P=�t�=��T=�{=�x�=�/=�;d>z�B
U�B��B�,B��Bq�B�[B}*B[
A�8B
�B&�B�B0�B��B7 BG�B!  B��B�yB�\B�uB�B�B
#�A�dB&�B��BZVB��B �Bw�B��B2�B$V�B#R"B�B)�B+_B�B�B��BS�A�HB�6B4�BqnB�B!�B$�2B,d�BB�B@�B/��B��A�|�B�CB$��BB~�BJ�B-ZB!�B� BOB:wB
=�B�{B�B��B?6B��BD8BS�A��B	�7B@/B��B18�B��B�>B>rB �NB��B�1BAuBLmB4eB��B	�RA���B:�BD�B�uB�
B@�B��BB2�B$@B#;�B�B*5fB�B��B�ZB��B@A�}@B�0B@B�9B3�B!��B%8_B,~�BB@BA�B/��B�UA�~yB�tB$��B��B��B[�B;>B �BA�BDHBA3A���A��A�eR?cW�C���A��)=^,�A�mA�NXA�dA��A=�SAf.PAS]jA]l�AmNA.�:A�}�A�ԥA��A���A��C@�;�A�?FA�}AKI6AH�A=�A���A�f9@�f?A@%YATl@F��@��QA���@m"�A�*�A��A[�;A��C���A�#�AӓB>��aA���A��A�LA��A� ?��@��'ADVA��A�G�C�AL��@�A{�[A��(@�@�=�Aa�G@���C���A�lA��WA�o�?iq�C��7A���=Co2Aև2A��A��cA�nA=�Ag�AS�A^��Aq�@A-�"A��A�bA�p�A��A�z@��A���A�EAJ��AIpA>N�A��A�xY@�-�AB�AT��@DM @���A�}W@f�TAꕜA�*LAZ�AƀC��A�NAӌ�?53AʁmA��jA�~�A!�AN`?���@��A�pA�a�A��C�@AL��@�A|�TA�}@�%o@���Aa�@��C���               	   
      �                        J   +         4      6      J                        8      �      :      N                     #   h         	         !         
      	         
      +      
   )                        9                        K   )         1      !      3                        -      -      +      9                     !   1                                                                                 !                        1                        -                        %            '      1                        !                                                         N��Ni��MѠ�N$��NS�N���N���O�_�N��_N��CN%O|NŊzN�d{N� �N5�\P8��Oa:�N^'�N
O���OE��O��$N4<P<WN��N�O��N��NCs�N���O!F�P��NO_jO�N�'9P&/�OKPI��N+OF-HN8�OAOe�N�%�O���O�tN���N`�qN�C�NH�N}~O"V�N��N)oN�N���N�K�N�4N���N.	�Nl^O&j%N�-�N6Q�N�1�  1  �  |    N      �  �  ?  d  
  �  �  �  �  1  �  e  �  5  @  �  �  �  �    7  N  �  +  �  �  �  �    �  .  �    �  W  H    <  �  �  
    {  u    �  ?  c  �    y     /  �  	[    r  ��o�o��o��o�u�T���D��=@���`B��o;�`B<#�
;��
<t�<o=�P<�`B<D��<D��=t�<��
=�P<��
<�<�j<�j<�j<���<���<���<ě�=t�<���=��`<�=C�<�h='�<��=o=\)=\)=�P='�=,1=��P=#�
='�=#�
=49X=H�9=P�`=H�9=T��=y�#=y�#=y�#=�7L=�C�=�t�=�t�=��w=��=���=�
=ddgt�����������|togd(!!)168>>;6)((((((((YVY[\gikg[YYYYYYYYYY��������������������aUKH=<::<HSUaaaaaaaa����������������������{}����������������"$5BN[gkw||{tg[F)""$/4;ECB;/"eglt�������tngeeeeee� !��������VUW[^htwz{xtihe[VVVV2-46CO\ab\[QODCB7622XSR[^hptvxutlhd[XXXX}�������������}}}}}}����)-@EDA5)������������������������������������")6?BFCB96+)""""""""��������
 /6/)
���#+/8<HPRQMG</#�������
������#%).6;BCBA=62)######@::ANgt����������g[@335;HT`\UTH;33333333 �
##$&&#
    TUW\anyz|��zna[VUTT��������������������@ABO[^`_[OKB@@@@@@@@��������������������ahjt����������tpnjha~yyvx��������������~RP\hu}wuh\RRRRRRRRRR�����
$&%#
���������� �����������������������������rr{����������������r������&)),+'���$%),+)!��������������
"
���������|{���������������||"/;HTWYVPHC;3/" #'/<?HIPSLH<92/+#  ����������������������������������������./7<HRUW^UTHA<4/....�������������������� 
#(.+(#
      ��������������������cehpt�����thccccccccab``aeht��������tnha776*$*-777777777UUUHGDHHMU[[UUUUUUUU]XYamxz|}|zmfa]]]]]]������

����������
"#)*$#
),1.)0066BBO[db[OMB:60000����������������������������������������)6BCJGA62)#!!"""#//3;<<<:5/###wnpyz|����zwwwwwwww���������������������
�������
����������������������
�����������������������������������������������������������������������������������������������������E\EXEPEMEPEPE\EiEoEuEiEdE\E\E\E\E\E\E\E\���������������������{�|���������������������������ùƹϹԹѹϹù�����������������)�6�B�T�]�`�]�T�O�6��������������a�b�j�m�v�z���z�m�a�]�[�\�Z�a�a�a�a�a�a�T�V�a�e�f�a�^�T�H�C�@�F�H�P�T�T�T�T�T�Tù��������������ÿùðòùùùùùùùù�A�M�Z�c�f�p�k�f�Z�M�A�<�4�4�4�6�A�A�A�A�G�T�`�b�i�k�d�`�T�H�G�;�6�.�&�.�;�<�G�G�ʾ׾��������׾ʾȾ������žʾʾʾʿ	���"�.�/�.�)�"����	��	�	�	�	�	�	���ſ׿ͿͿĿ������m�`�G�;�0�.�4�=�T�����н������)�4�8�4�,���ݽĽ������Ľ��5�A�N�N�T�S�N�A�<�5�*�0�5�5�5�5�5�5�5�5ÓÝÜàãàÓÑÇÄÇÉÓÓÓÓÓÓÓÓ���	��"�+�/�4�6�2�'�"��	��������������g�s�����������������������������q�e�b�g�)�5�B�N�[�]�Z�N�B�5�)�������
���)�ûлֻܻ����߻ܻлǻû��ûûûûû��5�A�N�g�x�������g�O�5����������5���(�0�*�0�(������������������������ɾ���������������������������������������������������~�x�x����������A�M�Z�f�s�j�i�f�[�Z�M�K�A�5�4�3�4�<�A�A�z�����������z�m�l�k�m�v�z�z�z�z�z�z�z�zčĚĦħıĲĦĚčĈăĄčččččččč�'�1�4�:�9�4��������ܻѻ������'�A�M�f��������������s�f�Z�9�&��&�-�4�A�׾�����׾־оо׾׾׾׾׾׾׾׾׾׺ֺ�������������ֺӺɺǺźɺֻ̺������������������x�l�b�k�l�x�������������������������������g�Y�L�F�C�G�O�Z�s���������!�-�2�:�I�F�:�!������������<�I�U�h�{ŊŐŋ�{�n�<�������������$�<�/�<�H�J�H�<�9�/�#��#�(�/�/�/�/�/�/�/�/����	��"�.�;�E�;�9�.�(�	����ݾݾ����a�g�n�o�s�n�a�^�U�M�U�X�a�a�a�a�a�a�a�aE�E�FFFF F$F%F$FFE�E�E�E�E�E�E�E�E��
��#�*�-�1�/�'�#��
�����������������
����������������������������Ϲܺ�
�������ܹϹù���������������Óàìù��������ûïàÓÇ�z�i�e�g�vÇÓ�z�{ÇËÏÓÖÓÇ�|�z�r�n�j�n�x�z�z�z�zìù��������ûùìàÝàãéìììììì������	�������ڼڼ�������ｅ�����������y�w�x�v�y�������������������L�Y�\�e�l�g�e�Y�L�H�A�B�L�L�L�L�L�L�L�L�:�F�S�_�l�x�{�{�l�_�S�O�F�?�:�/�,�-�.�:�x�l�l�i�i�l�y�}�������x�x�x�x�x�x�x�x�x�5�5�5�?�B�N�[�\�[�Y�N�B�5�5�5�5�5�5�5�5ŭŹ����������ŹŭŠŜřŠŦŭŭŭŭŭŭD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Dƾ������žƾ����������������������������������������������������������������������Ŀѿܿݿ�����ݿڿҿѿɿĿ��ĿĿĿ��)�6�8�6�0�6�@�6�)�����$�)�)�)�)�)�)��������������������������������������������������������������������}������������"�.�;�G�T�Z�T�T�G�;�6�.�(�"������M�Y�f�r�t�r�j�f�Y�S�M�K�M�M�M�M�M�M�M�MDoD{D�D�D�D�D�D�D�D{DoDkDhDnDoDoDoDoDoDo m 4 L G ` Z r 6 { ; > ( 3 # ] [ ~ 8 ] B Q   p C 5 , + $ . 8 m ! a  ` 5 � @ k L b 6 + O b 7 b ? 9 v 5 7 A I G b /   w Q R J k $ %    C  w  
  [  �    �  ,  �  �  R  �    �  z  �  �  n  _  =  �    �  I  �  �  %    ^  �  �  \  �  ;  �  �  �  �  :  �  y  5  �  �  m  �  �  l  �  �  �  f  =  L  �  �  �  #  �  ^  �  �  �  Q  �  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  1  1  0  ,  #      �  �  �  �  x  J    �  �  �    F    �  �  �  �  �  �  �  �  �  ~  v  i  \  O  B  /       �   �  |  x  s  n  i  d  Z  P  G  =  %  �  �  �  �  h  L  0    �                                      &  /  N  P  S  Z  e  m  j  f  b  ]  S  D  5  %      �  �  �  �      
    �  �  �  �  �  �  |  K    �  �  �  _  ,   �   �    �  �  �  �  �  �  �  �  �  �  �  �    A      �  )  �  
  R    �  �  Z  �  �  �  �  g    �  �      
�  	  �  7  �  �  �  |  p  f  _  X  R  K  J  M  Q  T  X  X  X  X  W  W  ?  3  '        �  �  �  �  �  �  w  d  Q  >    �  �  �  -  U  b  ]  V  M  <  &    �  r    �  ]  �  �  ,  �  P   �  �  �  �  �    	  	    �  �  �  �  �  �  �  y  G    �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  b  D     �   �   �   O  g  y  �  �  �  �  �  �  z  m  _  N  <  )      �  �  �  �  �  �  �  �  �  �  �             &  ,  2  8  0  %        E  j  �  �  �  �  �  �  �  �  k  )  �  t    {  �  �     �  �          '  -  0     	  �  �  �  6  t  �  �  �   �  �  �  �  �  �  �  �  �  �  {  i  V  <  #    �  �  �  t  M  e  a  ]  Y  V  R  N  F  <  2  '      	  �  �  �  �  �  �  @  �  �  �  �  �  �  �  �  �  �  �  �  �  p  6  �  A  {  �  1  5  .  %        �  �  �  �  �  k  F  %  �  �  �  +  �  $  o  �  �  !  9  ?  ?  3    �  �  x    �    r  �  ,  �  �  �  �  |  k  H  %    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  $  �  �  3  �  U  �  !  L  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  x  �  �  �  �  �  u  X  7    �  �  �  u  L  (    �  �  �  �        �  �  �  �  �  p  W  ;    �  �  �  �  {  T  �  
      &  6  #    �  �  �  �  �    c  G  *    �  �    1  <  D  J  M  N  K  B  )    �  �  �  M    �  �  M    �  l  l  �  �  r  \  E  +    �  �  �  �  X    �  W  �  5  �  �  +  !        �  �  �  �  �  g  C  �  �  �  V  �  u  �  0  8  Y  w  �  �  �  x  g  V  =    	  �  �  �  h  %  �  �  �  �  �  �  �  �  �  �    |  x  u  q  m  k  i  h  g  f  e  c  	�  
�  �  3  �  M  �    [    �  j     �  ?  �  h  	P  �  �  R  4  z  �  �  w  Z  3  	  �  �  }  A  �  �  >  �  �  p  m  �      	  �  �  �  �  L  �  �  E  �  s  �  �     �  �  �  �  �  �  �  �  �  q  U  6    �  �  �  �  �  �  k  G  �  �    �  *  #    �  �  �  �  �  o  C  a  J    �  �    J  3  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      0  A    	    �  �  �  �  �  �  �  s  Z  ;    �  �  �  g  N  F  �  �  �  �  �  �  �  t  a  Z  U  M  3    �  �  �  @  �  �  W  R  H  ;  +      �  �  �  �  �  �  j  I  $  �    �  I  H  =  <  ?  )    �  �  �  �  S    �  �  3  �  `  �  �     �            �  �  �  �  ^  /  �  �  �  B  �  �  I  �    /  8  <  4  #  
  �  �  �  U    �  m    �  +  s  �   �  
V  J  �  
  J  p    w  R    �  /  
�  
  	G  ]    L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      4  �         �  �  �  �  �  �  k  @  �  d    �  �  D  �  �    �  �  �  �  �  �  �  �  w  i  [  N  A  5  %      *  A  z  z  {  {  u  n  f  Y  H  8  &    �  �  �  �  W  %   �   �  n  r  t  u  t  o  d  P  2    �  �  �  o  6  �  �  x  >            �  �  �  �  �  �  �  q  ;  �  �    Y  �  �  3  �  �  �  �  �  �    s  f  Y  L  ?  2  %       �   �   �   �  ?  :  4  -  $      	    �  �  �  �  �  �  �  w  4  �  �  c  <    �  �  �  �  b  +  �  �  �  �  h  N  4    �  �  �  �  �  �  �  c  =    �  �  �  a  -  �  �  �    w  �  �  6    �  �  �  �  �  �  y  f  T  @  )    �  �  �  }  >   �   �  ?  =  A  U  c  l  s  z  }  ~  x  o  b  V  I  ;  -      �             �  �  �  �  �  �  �  �  �  �  �  �  p  W  >  /  .  .  (  "        �  �  �  �  �  v  b  O  >  *  �  �  �  �  �  l  U  8    �  �  �  }  Y  6    �  �  �  �  �    	$  	;  	P  	Z  	E  	  �  �  �  W     �  �  =  �  @  d  _  H  �    
  �  �  �  �  �  ~  L    �  �  �  R    �  �  X    �  r  d  V  E  4     	  �  �  �  �  j  8    �  �  c  -  �  �  �  }  d  E  "  �  �  v  "  
�  
X  	�  	k  �  A  �  �     �  �